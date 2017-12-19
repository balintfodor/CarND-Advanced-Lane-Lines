import glob
import sys
import copy
from tqdm import tqdm

import numpy as np
import cv2
from consecution import Pipeline, Node, GroupByNode
from skimage.exposure import equalize_adapthist
from scipy.optimize import least_squares
from scipy.ndimage.measurements import center_of_mass

class Bypass(Node):
    def process(self, item):
        self.push(item)


class ImageLog(Node):
    def __init__(self, name, base_dir):
        super().__init__(name)
        self.base_dir = base_dir

    def process(self, item):
        item = copy.deepcopy(item)
        image = item['image']
        path = '{}/{}-{:03d}.png'.format(self.base_dir, self.name, im_id)
        cv2.imwrite(path, image)
        self.push(item)


class Undistort(Node):
    def __init__(self, name, camera_matrix, camera_distortion, image_size):
        super().__init__(name)
        self.matrix = camera_matrix
        self.distortion = camera_distortion
        self.image_size = image_size
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            self.matrix, self.distortion, None, None,
            tuple(self.image_size), self.distortion.size)

    def process(self, item):
        item = copy.deepcopy(item)
        image = item['image']
        und = cv2.remap(image, self.mapx, self.mapy, cv2.INTER_LINEAR)
        item['image'] = und
        self.push(item)


class SegmentLines(Node):
    def process(self, item):
        item = copy.deepcopy(item)
        image = item['image']
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for i in range(3):
            image[:, :, i] = clahe.apply(image[:, :, i])

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        bin_yellow = np.logical_and(np.logical_and(h > 15, h < 35), v > 190)
        bin_white = np.logical_and(s < 30, v > 200)

        sobel_x = cv2.Sobel(s, cv2.CV_64F, 1, 0, ksize=15)
        sobel_y = cv2.Sobel(s, cv2.CV_64F, 0, 1, ksize=15)

        dir_min = 0.4 * np.pi / 2
        dir_max = 0.8 * np.pi / 2

        grad_dir = np.arctan2(np.abs(sobel_y), np.abs(sobel_x))
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        grad_mag = grad_mag / np.max(grad_mag) * 255

        bin_grad_dir = np.logical_and(grad_dir > dir_min, grad_dir < dir_max)
        bin_grad_mag = grad_mag > 5

        color_bin = np.logical_or(bin_yellow, bin_white)
        bin_grad = np.logical_and(bin_grad_dir, bin_grad_mag)
        bin_out = np.logical_and(bin_grad, color_bin)

        item['image'] = (bin_out * 255).astype(np.uint8)
        self.push(item)


class Add(GroupByNode):
    def key(self, item):
        return item['id']

    def process(self, batch):
        images = [item['image'] for item in batch]
        out = np.sum(np.array(images), axis=0)
        self.push({'id': batch[0]['id'], 'image': out})


class PerspectiveWarp(Node):
    def __init__(self, name, src_points, dst_points, src_size, dst_size):
        super().__init__(name)
        self.src_pts = np.array(src_points, dtype=np.float32)
        self.dst_pts = np.array(dst_points, dtype=np.float32)
        self.matrix = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        self.dst_size = tuple(dst_size)
        self.src_size = tuple(src_size)
        self.inverse = False

    def process(self, item):
        item = copy.deepcopy(item)
        image = item['image']
        if self.inverse:
            out = cv2.warpPerspective(
                image,
                self.matrix,
                self.src_size,
                flags=cv2.WARP_INVERSE_MAP)
        else:
            out = cv2.warpPerspective(image, self.matrix, self.dst_size)
        item['image'] = out
        self.push(item)


class LaneDetector(Node):
    def generate_window_list(self, im, width, x_offset, height, y_step):
        win_list = []
        offset_list = []
        for y in range(im.shape[1] - height, 0, -y_step):
            win_list.append(im[y:(y + height), x_offset:(x_offset + width)])
            offset_list.append((y, x_offset))
        return win_list, np.array(offset_list)

    def select_points(self, roi_list, offset_list):
        p = []
        for i, roi in enumerate(roi_list):
            c = center_of_mass(roi)
            if np.any(np.isnan(c)):
                continue
            p.append(c + offset_list[i])
        return p


    def fit_quadratic(self, points):
        def f(p, x, y, w):
            e1 = np.polyval(p[[0, 1, 2]], x) - y
            e2 = np.polyval(p[[0, 3, 4]], x) - y
            return np.sqrt(w) * np.minimum(np.abs(e1), np.abs(e2))

        weights = np.floor(points[:, 1] / 64) + 1

        p0 = [1, 1, 1, 0, 0]
        result = least_squares(f, p0, args=(points[:, 0], points[:, 1], weights))
        return result.x

    def process(self, item):
        item = copy.deepcopy(item)
        image = item['image']

        win_height = 16
        win_y_step = 16
        im_half_w = image.shape[1] // 2

        wl1, ol1 = self.generate_window_list(image, im_half_w, 0, win_height, win_y_step)
        wl2, ol2 = self.generate_window_list(image, im_half_w, im_half_w, win_height, win_y_step)
        point_set1 = self.select_points(wl1, ol1)
        point_set2 = self.select_points(wl2, ol2)

        points = []
        points.extend(point_set1)
        points.extend(point_set2)

        if len(points) > 1:
            points = np.array(points)
            p = self.fit_quadratic(points)
            lanes = [[p[0], p[1], p[2]], [p[0], p[3], p[4]]]
            lane_points = [point_set1, point_set2]
        else:
            lanes = [None, None]
            lane_points = [[], []]

        item['lanes'] = lanes
        item['lane_points'] = lane_points

        self.push(item)


class PolyImageLog(Node):
    def __init__(self, name, base_dir):
        super().__init__(name)
        self.base_dir = base_dir

    def process(self, item):
        item = copy.deepcopy(item)
        im_id, image, lanes, lane_points = item['id'], item['image'], item['lanes'], item['lane_points']
        # im_id, image, lane_points = item['id'], item['image'], item['lane_points']
        samples = range(image.shape[0])
        out = np.copy(image) // 2
        for k, poly in enumerate(lanes):
            self.draw_poly(out, poly, samples)
            for y, x in lane_points[k]:
                cv2.circle(out, (int(x), int(y)), 2, (255,))

        path = '{}/{}-{:03d}.png'.format(self.base_dir, self.name, im_id)
        cv2.imwrite(path, out)
        self.push(item)

    def draw_poly(self, im, p, xs):
        x = xs[0]
        y = np.polyval(p, x)
        for x_new in xs[1:]:
            y_new = np.polyval(p, x_new)
            cv2.line(im, (int(y), int(x)), (int(y_new), int(x_new)), (255,))
            y = y_new
            x = x_new


class MaskWithPoly(Node):
    def __init__(self, name, width=48, win_height=32):
        super().__init__(name)
        self.half_width = width / 2
        self.win_height = win_height

    def process(self, item):
        item = copy.deepcopy(item)
        im_id, image, lanes = item['id'], item['image'], item['lanes']
        wh = self.win_height
        out = np.zeros_like(image)
        for k, poly in enumerate(lanes):
            for y_pos in range(0, 255, wh):
                x = quadratic(poly, y_pos - wh / 2)
                x_min = np.maximum(0, x - self.half_width)
                x_max = np.minimum(out.shape[1], x + self.half_width)
                x_min, x_max = int(x_min), int(x_max)
                out[y_pos:(y_pos + wh),
                    x_min:x_max] = image[y_pos:(y_pos + wh),
                                         x_min:x_max]
        item['image'] = out
        self.push(item)


class LanePainter(Node):
    def process(self, item):
        item = copy.deepcopy(item)
        im_id, image, lanes = item['id'], item['image'], item['lanes']
        out = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        self.draw_lane(out, lanes[0], lanes[1])
        item['image'] = out
        self.push(item)

    def draw_lane(self, im, p1, p2):
        points = [(quadratic(p1, y), y) for y in range(im.shape[0])]
        points.extend([(quadratic(p2, y), y)
                       for y in reversed(range(im.shape[0]))])
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(im, [pts], (0, 64, 0))


class InfoPainter(Node):
    def __init__(self, name, lane_width_in_pixels, lane_width_in_m=3.7):
        super().__init__(name)
        self.width_p2m = lane_width_in_m / lane_width_in_pixels

    def process(self, item):
        item = copy.deepcopy(item)
        im_id, image, lanes = item['id'], item['image'], item['lanes']
        sz = [256, 256]
        offset = self.car_offset(sz, lanes) * self.width_p2m
        r = np.mean([
            self.curv_radius(
                lanes[np.random.randint(2)],
                np.random.rand() * (sz[0] - 1)) for _ in range(100)
        ])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            image, 'offset = {:0.2f} m, curvature radius = {:0.2f} m'.format(
                offset, r), (10, 50), font, 1, (255, 255, 255), 2)

        item['image'] = image
        self.push(item)

    def car_offset(self, im_size, lanes):
        x1 = quadratic(lanes[0], im_size[0] - 1)
        x2 = quadratic(lanes[1], im_size[0] - 1)
        c = 0.5 * (x1 + x2)
        return c - im_size[1] / 2

    def curv_radius(self, p, h):
        return np.power(
            1 + np.square(2 * p[0] * h + p[1]), 3 / 2) / np.abs(2 * p[0])


def collect_images(folder):
    images = []
    for format in ['jpg', 'png', 'tif']:
        images.extend(glob.glob("{}/*.{}".format(folder, format)))
    return images


def image_loader(images):
    for i, im_path in enumerate(images):
        im = cv2.imread(im_path)
        item = {'id': i, 'image': im}
        yield item


def video_loader(video_file):
    cap = cv2.VideoCapture(video_file)
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        item = {'id': i, 'image': frame}
        i += 1
        yield item


def main():
    calib_file = 'debug/calibration.npz'
    calib = np.load(calib_file)

    input_dir = 'test_images'
    persp_src_points = [[137.9, 719], [
        558.3, 426.7], [674.9, 426.7], [1140.2, 719]]
    # persp_dst_points = [[137.9/1280*256, 270], [137.9/1280*256, 0], [1140.2/1280*256, 0], [1140.2/1280*256, 270]]
    persp_dst_points = [[30, 270], [30, 0], [225, 0], [225, 270]]
    persp_dst_size = [256, 256]

    undistort = Undistort(
        'undistort',
        calib['camera_matrix'],
        calib['distortion_coefs'],
        calib['image_size'])

    forward_warp = PerspectiveWarp('perspective_warp',
                                   persp_src_points,
                                   persp_dst_points,
                                   calib['image_size'],
                                   persp_dst_size)
    backward_warp = copy.deepcopy(forward_warp)
    backward_warp.name = 'perspective_back_warp'
    backward_warp.inverse = True

    rough_lane_detector = LaneDetector('lane_detector_rough')
    finer_lane_detector = LaneDetector(
        'lane_detector_finer',
        use_argmax_for_center_detect=False)

    pipe = Pipeline(
        undistort
        | SegmentLines('color_threshold')
        | forward_warp
        | LaneDetector('lane_detector')
        | PolyImageLog('lane_painter_log', 'output_images')
    )

    print(pipe)

    # images = collect_images(input_dir)
    # pipe.consume(tqdm(image_loader(images), total=len(images)))

    pipe.consume(tqdm(video_loader('project_video.mp4')))
    # pipe.consume(tqdm(video_loader('challenge_video.mp4')))
    # pipe.consume(tqdm(video_loader('harder_challenge_video.mp4')))


if __name__ == "__main__":
    main()
