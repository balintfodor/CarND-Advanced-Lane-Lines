import cv2
import numpy as np
from scipy.optimize import least_squares
from consecution import Pipeline, Node, GroupByNode
import glob
import copy
from tqdm import tqdm

class Bypass(Node):
    def process(self, item):
        self.push(item)


class ImageLog(Node):
    def __init__(self, name, base_dir):
        super().__init__(name)
        self.base_dir = base_dir

    def process(self, item):
        item = copy.deepcopy(item)
        im_id, image = item['id'], item['image']
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
        im_id, image = item['id'], item['image']
        und = cv2.remap(image, self.mapx, self.mapy, cv2.INTER_LINEAR)
        item['image'] = und
        self.push(item)


class ColorThreshold(Node):
    def __init__(self, name, sat_th, val_th):
        super().__init__(name)
        self.sat_th = sat_th
        self.val_th = val_th

    def process(self, item):
        item = copy.deepcopy(item)
        im_id, image = item['id'], item['image']
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        bin_image = (hsv[:, :, 1] >= self.sat_th) * (hsv[:, :, 2] >= self.val_th)
        out = (bin_image * 255).astype(np.uint8)
        item['image'] = out
        self.push(item)


class GradientThreshold(Node):
    def __init__(self, name, mag_limits, dir_limits, kernel_size=5):
        super().__init__(name)
        self.kernel_size = kernel_size
        self.mag_limits = mag_limits
        self.dir_limits = [
            np.maximum(dir_limits[0], 0) * np.pi/2, 
            np.minimum(dir_limits[1], 1) * np.pi/2]

    def process(self, item):
        item = copy.deepcopy(item)
        im_id, image = item['id'], item['image']
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.kernel_size)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.kernel_size)

        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        grad_mag = grad_mag / np.max(grad_mag) * 255

        grad_dir = np.arctan2(np.abs(sobel_y), np.abs(sobel_x))

        bin_image = (grad_mag >= self.mag_limits[0]) * (grad_mag < self.mag_limits[1]) \
            * (grad_dir >= self.dir_limits[0]) * (grad_dir < self.dir_limits[1])

        out = (bin_image * 255).astype(np.uint8)
        item['image'] = out
        self.push(item)


class Merger(GroupByNode):
    def __init__(self, name, method='max'):
        super().__init__(name)
        self.method = method

    def key(self, item):
        return item['id']

    def process(self, batch):
        images = [item['image'] for item in batch]
        if self.method == 'max':
            out = np.max(np.array(images), axis=0)
        else:
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
        im_id, image = item['id'], item['image']
        if self.inverse:
            out = cv2.warpPerspective(image, self.matrix, self.src_size, flags=cv2.WARP_INVERSE_MAP)
        else:
            out = cv2.warpPerspective(image, self.matrix, self.dst_size)
        item['image'] = out
        self.push(item)

def quadratic(coeffs, x):
    return coeffs[0] * x*x + coeffs[1] * x + coeffs[2]

class LaneDetector(Node):
    def __init__(self, name, win_height=32, overlap_param=4, use_argmax_for_center_detect=True):
        super().__init__(name)
        self.win_height = win_height
        self.use_argmax = use_argmax_for_center_detect
        self.overlap_param = int(overlap_param)

    def analyze_win(self, im, last_x):
        vert_sum = np.sum(im, axis=0)
        if last_x is not None:
            mask = np.zeros_like(vert_sum)
            b = np.maximum(0, int(last_x)-16)
            e = np.minimum(im.shape[1]-1, int(last_x)+16)
            mask[b:e] = 1
            vert_sum *= mask
        s = np.sum(vert_sum)
        if s < self.win_height * 2 * 255:
            return None

        if self.use_argmax:
            x = np.argmax(vert_sum)
        else:
            idx = range(im.shape[1])
            val = vert_sum / s
            x = np.sum(idx * val)
        return x

    def robust_poly_pair_fit(self, points1, points2):
        points1 = np.array(points1)
        points2 = np.array(points2)
        # points = np.vstack((points1, points2))

        def f(p, x, y):
            e1 = quadratic(p[[0, 1, 2]], x) - y
            return e1
        
        p0 = np.array([1, 1, 1])
        res1 = least_squares(f, p0, loss='soft_l1', f_scale=0.1,
            args=(points1[:, 0], points1[:, 1]))

        res2 = least_squares(f, p0, loss='soft_l1', f_scale=0.1,
            args=(points2[:, 0], points2[:, 1]))

        return ([res1.x[0], res1.x[1], res1.x[2]], [res2.x[0], res2.x[1], res2.x[2]])

    def process(self, item):
        item = copy.deepcopy(item)
        im_id, image = item['id'], item['image']
        wh = self.win_height
        half = image.shape[1] // 2
        step = wh // self.overlap_param
        h_samples = range(255 - wh, 0, -step)

        lanes = [None, None]
        lane_points = [None, None]
        for k, side_offset in enumerate([0, half]):
            points = []
            last_x = None
            for y_pos in h_samples:
                roi = image[y_pos:(y_pos + wh), side_offset:(side_offset + half)]
                x = self.analyze_win(roi, last_x)
                if x is not None:
                    y = y_pos + wh / 2
                    points.append((y, x + side_offset))
                    last_x = x
            lane_points[k] = points

        lanes = self.robust_poly_pair_fit(lane_points[0], lane_points[1])

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
        y = quadratic(p, x)
        for x_new in xs[1:]:
            y_new = quadratic(p, x_new)
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
                x = quadratic(poly, y_pos - wh/2)
                x_min = np.maximum(0, x - self.half_width)
                x_max = np.minimum(out.shape[1], x + self.half_width)
                x_min, x_max = int(x_min), int(x_max)
                out[y_pos:(y_pos + wh), x_min:x_max] = image[y_pos:(y_pos + wh), x_min:x_max]
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
        points.extend([(quadratic(p2, y), y) for y in reversed(range(im.shape[0]))])
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
                np.random.rand()*(sz[0]-1)) for _ in range(100)
            ])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image,
            'offset = {:0.2f} m, curvature radius = {:0.2f} m'.format(offset, r),
            (10, 50), font, 1, (255, 255, 255), 2)

        item['image'] = image
        self.push(item)
    
    def car_offset(self, im_size, lanes):
        x1 = quadratic(lanes[0], im_size[0]-1)
        x2 = quadratic(lanes[1], im_size[0]-1)
        c = 0.5 * (x1 + x2)
        return c - im_size[1] / 2

    def curv_radius(self, p, h):
        return np.power(1 + np.square(2 * p[0] * h + p[1]), 3/2) / np.abs(2 * p[0])

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
    persp_src_points = [[137.9, 719], [558.3, 426.7], [674.9, 426.7], [1140.2, 719]]
    persp_dst_points = [[137.9/1280*256, 265], [137.9/1280*256, 0], [1140.2/1280*256, 0], [1140.2/1280*256, 265]]
    # persp_dst_points = [[10, 265], [10, 0], [245, 0], [245, 265]]
    persp_dst_size = [256, 256]

    forward_warp = PerspectiveWarp('perspective_warp', 
        persp_src_points, 
        persp_dst_points, 
        calib['image_size'], 
        persp_dst_size)
    backward_warp = copy.deepcopy(forward_warp)
    backward_warp.name = 'perspective_back_warp'
    backward_warp.inverse = True

    rough_lane_detector = LaneDetector('lane_detector_rough')
    finer_lane_detector = LaneDetector('lane_detector_finer', use_argmax_for_center_detect=False)

    pipe = Pipeline(ImageLog('original', 'output_images')
        | Undistort('undistort', calib['camera_matrix'], calib['distortion_coefs'], calib['image_size']) | ImageLog('undistorted', 'output_images')
        | [
            [
                ColorThreshold('color_threshold', 80, 200) | ImageLog('color_threshold_log', 'output_images'),
                GradientThreshold('gradient_threshold', [15, 250], [0.3, 0.9]) | ImageLog('gradient_threshold_log', 'output_images')
            ]
            | Merger('max_merger', 'max') | ImageLog('max_merger_out', 'output_images')
            | forward_warp | ImageLog('perspective_warp_out', 'output_images')
            | rough_lane_detector | PolyImageLog('poly_log', 'output_images')
            | MaskWithPoly('mask_with_poly') | ImageLog('mask_with_poly_out', 'output_images')
            | finer_lane_detector | PolyImageLog('poly_refined_log', 'output_images')
            | LanePainter('lane_painter') | PolyImageLog('lane_painter_log', 'output_images')
            | backward_warp | ImageLog('perspective_back_warp_out', 'output_images')
            | InfoPainter('info_painter', 188) | ImageLog('info_painter_out', 'output_images')
            ,
            Bypass('undist_bypass')
        ]
        | Merger('add_merger', 'add') | ImageLog('final_image', 'output_images')
    )

    # pipe = Pipeline(
    #     Undistort('undistort', calib['camera_matrix'], calib['distortion_coefs'], calib['image_size'])
    #     | [
    #         [
    #             ColorThreshold('color_threshold', 80, 200),
    #             GradientThreshold('gradient_threshold', [50, 250], [0.3, 0.9])
    #         ]
    #         | Merger('max_merger', 'max')
    #         | forward_warp
    #         | rough_lane_detector
    #         | MaskWithPoly('mask_with_poly')
    #         | finer_lane_detector
    #         | LanePainter('lane_painter')
    #         | backward_warp
    #         | InfoPainter('info_painter', 188)
    #         ,
    #         Bypass('undist_bypass')
    #     ]
    #     | Merger('add_merger', 'add') | ImageLog('final_image', 'output_images')
    # )

    print(pipe)

    # images = collect_images(input_dir)
    # pipe.consume(tqdm(image_loader(images), total=len(images)))

    pipe.consume(tqdm(video_loader('challenge_video.mp4')))

if __name__ == "__main__":
    main()