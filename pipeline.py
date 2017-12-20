import glob
import sys
import copy
from tqdm import tqdm

import numpy as np
import cv2
from consecution import Pipeline, Node, GroupByNode
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
        image, im_id = item['image'], item['id']
        path = '{}/{:03d}-{}.png'.format(self.base_dir, im_id, self.name)
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


class LineScoreMap(Node):
    def color_score(self, hsv, target_hsv, max_dist):
        d = np.linalg.norm(hsv - target_hsv, axis=2)
        m = 1 - np.minimum(d, max_dist) / max_dist
        return m

    def grad_map(self, im):
        sobel_x = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=5)
        grad_dir = np.arctan2(np.abs(sobel_y), np.abs(sobel_x))
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        grad_mag = grad_mag / np.max(grad_mag)
        return grad_dir, grad_mag

    def process(self, item):
        item = copy.deepcopy(item)
        image = item['image']

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        score = np.zeros_like(h, dtype=np.float)

        s_grad_dir, s_grad_mag = self.grad_map(s)
        v_grad_dir, v_grad_mag = self.grad_map(v)

        dv = np.linalg.norm(v_grad_dir[:, :, np.newaxis] - 0.6, axis=2)
        ds = np.linalg.norm(s_grad_dir[:, :, np.newaxis] - 0.6, axis=2)

        score += v_grad_mag
        score += s_grad_mag
        score += (1 - np.minimum(dv, 0.2) / 0.2) * v_grad_mag
        score += (1 - np.minimum(ds, 0.2) / 0.2) * s_grad_mag

        yellow = [41 * 255 / 360, 60 * 255 / 100, 97 * 255 / 100]
        dark_yellow = [39 * 255 / 360, 46 * 255 / 100, 41 * 255 / 100]
        white = [36 * 255 / 360, 8 * 255 / 100, 99 * 255 / 100]

        score += self.color_score(hsv, yellow, 200)
        score += self.color_score(hsv, dark_yellow, 50)
        score += self.color_score(hsv, white, 50)

        score = score / np.max(score)

        item['image'] = (score * 255).astype(np.uint8)
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
    def begin(self):
        self.prev_points = np.zeros(shape=(0, 3))
        self.p0 = [0, 0, 0, 255]

    def end(self):
        pass

    def generate_window_list(self, im, width, x_offset, height, y_step):
        win_list = []
        offset_list = []
        for y in range(im.shape[1] - height, 0, -y_step):
            win_list.append(im[y:(y + height), x_offset:(x_offset + width)])
            offset_list.append((y, x_offset))
        return win_list, np.array(offset_list)

    def select_points(self, roi_list, offset_list, sub_win=16):
        p = []
        for i, roi in enumerate(roi_list):
            c = np.unravel_index(np.argmax(roi), roi.shape)
            r_min = np.maximum(0, int(c[0] - sub_win))
            r_max = np.minimum(roi.shape[0], int(c[0] + sub_win))
            c_min = np.maximum(0, int(c[1] - sub_win))
            c_max = np.minimum(roi.shape[1], int(c[1] + sub_win))
            sub_roi = roi[r_min:r_max, c_min:c_max]
            cm = center_of_mass(sub_roi)
            if np.any(np.isnan(cm)):
                p.append(c + offset_list[i])
            else:
                p.append(cm + offset_list[i] + c - sub_win)
        return p

    def fit_quadratic(self, points_with_weights, p0):
        def f(p, x, y, w):
            e1 = np.polyval(p[[0, 1, 2]], x) - y
            e2 = np.polyval(p[[0, 1, 3]], x) - y
            return np.sqrt(w) * np.minimum(np.abs(e1), np.abs(e2))

        pts = points_with_weights
        result = least_squares(f, p0, args=(pts[:, 0], pts[:, 1], pts[:, 2]), bounds=(
            [-0.0015, -np.inf, -np.inf, -np.inf], [0.0015, np.inf, np.inf, np.inf]),
            loss='soft_l1')
        return result.x, result.fun

    def process(self, item):
        item = copy.deepcopy(item)
        image = item['image']

        win_height = 16
        win_y_step = 16
        im_half_w = image.shape[1] // 2

        wl1, ol1 = self.generate_window_list(
            image, im_half_w, 0, win_height, win_y_step)
        wl2, ol2 = self.generate_window_list(
            image, im_half_w, im_half_w, win_height, win_y_step)
        point_set1 = self.select_points(wl1, ol1)
        point_set2 = self.select_points(wl2, ol2)

        points = []
        points.extend(point_set1)
        points.extend(point_set2)

        if len(points) > 4:
            points = np.array(points, ndmin=2)
            weights = np.floor(points[:, 0] / 64) * 2 + 1
            points = np.c_[points, weights]

            self.prev_points[:, 2] *= 0.5
            merged_points = np.vstack((points, self.prev_points))

            p, residuals = self.fit_quadratic(merged_points, self.p0)
            idx = np.argsort(residuals)
            self.prev_points = merged_points[idx[:(len(idx) // 2)], :]

            lanes = [[p[0], p[1], p[2]], [p[0], p[1], p[3]]]
            support_points = merged_points
        else:
            lanes = [[self.p0[0], self.p0[1], self.p0[2]],
                     [self.p0[0], self.p0[1], self.p0[3]]]
            support_points = self.prev_points

        item['lanes'] = lanes
        item['support_points'] = support_points

        self.push(item)


class LanePainter(Node):
    def process(self, item):
        item = copy.deepcopy(item)
        image, lanes, support_points = item['image'], item['lanes'], item['support_points']
        out = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        out[:, :, 2] = image
        out[:, :, 1] = self.draw_lane(out, lanes[0], lanes[1])
        out[:, :, 0] = self.draw_points(image, support_points)
        item['image'] = out
        self.push(item)

    def draw_lane(self, im, p1, p2):
        out = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
        points = [(np.polyval(p1, y), y) for y in range(im.shape[0])]
        points.extend([(np.polyval(p2, y), y)
                       for y in reversed(range(im.shape[0]))])
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(out, [pts], (64))

        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(out, '{:0.4f} {:0.4f} {:0.4f}'.format(
        #     *p1), (30, 50), font, 0.4, (255, 255, 255), 1)
        # cv2.putText(out, '{:0.4f} {:0.4f} {:0.4f}'.format(
        #     *p2), (30, 100), font, 0.4, (255, 255, 255), 1)
        return out

    def draw_points(self, im, pts):
        out = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
        for y, x, _ in pts:
            cv2.circle(out, (int(x), int(y)), 3, (255,))
        return out


class InfoPainter(Node):
    def __init__(self, name, lane_width_in_pixels, lane_length_in_pixels,
                 lane_width_in_m=3.7, lane_length_in_m=40):
        super().__init__(name)
        self.width_p2m = lane_width_in_m / lane_width_in_pixels
        self.length_p2m = lane_length_in_m / lane_length_in_pixels

    def process(self, item):
        item = copy.deepcopy(item)
        im_id, image, lanes = item['id'], item['image'], item['lanes']
        sz = [256, 256]
        offset = self.car_offset(sz, lanes) * self.width_p2m

        new_lanes = [
            self.scale_curve(
                lanes[i],
                self.width_p2m,
                self.length_p2m,
                sz) for i in range(2)]
 
        r = np.mean([self.curv_radius(new_lanes[i], sz[0]) for i in range(2)])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, 'offset = {:0.2f} m'.format(offset), (10, 70), font, 2, (255, 255, 255), 2)
        cv2.putText(image, 'curvature radius = {:0.2f} m'.format(r), (10, 150), font, 2, (255, 255, 255), 2)

        item['image'] = image
        self.push(item)

    def car_offset(self, im_size, lanes):
        x1 = np.polyval(lanes[0], im_size[0] - 1)
        x2 = np.polyval(lanes[1], im_size[0] - 1)
        c = 0.5 * (x1 + x2)
        return c - im_size[1] / 2

    def scale_curve(self, p, x_scale, y_scale, im_size):
        pts = [(y, np.polyval(p, y)) for y in range(im_size[0])]
        pts = np.array(pts)
        pts[:, 0] = im_size[0] - 1 - pts[:, 0]
        return np.polyfit(pts[:, 0] * y_scale, pts[:, 1] * x_scale, 2)

    def curv_radius(self, p, h):
        return (1 + (2 * p[0] * h + p[1])**2)**1.5 / np.abs(2 * p[0])


def collect_images(folder):
    images = []
    for format in ['jpg', 'png', 'tif']:
        images.extend(glob.glob("{}/*.{}".format(folder, format)))
    return images


def image_loader(images, pipe):
    for i, im_path in tqdm(enumerate(images), total=len(images)):
        im = cv2.imread(im_path)
        item = {'id': i, 'image': im}
        pipe.consume([item])


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
    persp_dst_points = [[40, 265], [40, 0], [215, 0], [215, 265]]
    persp_dst_size = [256, 256]

    undistort = Undistort(
        'undistort',
        calib['camera_matrix'],
        calib['distortion_coefs'],
        calib['image_size'])

    forward_warp = PerspectiveWarp('warp',
                                   persp_src_points,
                                   persp_dst_points,
                                   calib['image_size'],
                                   persp_dst_size)
    backward_warp = copy.deepcopy(forward_warp)
    backward_warp.name = 'warp_back'
    backward_warp.inverse = True

    pipe = Pipeline(
        ImageLog('p00-orig_img', 'output_images')
        | undistort | ImageLog('p01-undistort_img', 'output_images')
        | [
            LineScoreMap('segment') | ImageLog('p02-line_score_img', 'output_images')
            | forward_warp | ImageLog('p03-warp_img', 'output_images')
            | LaneDetector('detect')
            | LanePainter('lane_paint') | ImageLog('p04-lane_img', 'output_images')
            | backward_warp
            | InfoPainter('info_paint', 170, 265),
            Bypass('undistort_bypass')]
        | Add('add')
        | ImageLog('p05-final', 'output_images')
    )

    print(pipe)

    images = collect_images(input_dir)
    image_loader(images, pipe)

    # pipe.consume(tqdm(video_loader('project_video.mp4')))
    # pipe.consume(tqdm(video_loader('challenge_video.mp4')))
    # pipe.consume(tqdm(video_loader('harder_challenge_video.mp4')))


if __name__ == "__main__":
    main()
