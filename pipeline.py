import cv2
import numpy as np
from scipy.optimize import least_squares
from consecution import Pipeline, Node, GroupByNode
import glob
import copy
from tqdm import tqdm

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


class MaxMerger(GroupByNode):
    def key(self, item):
        return item['id']

    def process(self, batch):
        images = [item['image'] for item in batch]
        out = np.max(np.array(images), axis=0)
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

    def analyze_win(self, im):
        vert_sum = np.sum(im, axis=0)
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

    def robust_fit_poly(self, points):
        points = np.array(points)

        def f(p, x, y):
            return quadratic(p, x) - y
        
        p0 = np.array([1, 1, 1])
        res = least_squares(f, p0, loss='soft_l1', f_scale=0.1,
            args=(points[:, 0], points[:, 1]))
        return res.x

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
            for y_pos in h_samples:
                roi = image[y_pos:(y_pos + wh), side_offset:(side_offset + half)]
                x = self.analyze_win(roi)
                if x is not None:
                    y = y_pos + wh / 2
                    points.append((y, x + side_offset))
            poly = self.robust_fit_poly(points)
            lanes[k] = poly
            lane_points[k] = points

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

def main():
    calib_file = 'debug/calibration.npz'
    calib = np.load(calib_file)

    input_dir = 'test_images'
    persp_src_points = [[137.9, 719], [558.3, 426.7], [674.9, 426.7], [1140.2, 719]]
    persp_dst_points = [[137.9/1280*256, 255], [137.9/1280*256, 0], [1140.2/1280*256, 0], [1140.2/1280*256, 255]]
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
    finer_lane_detector = LaneDetector('lane_detector_finer',
        win_height=16, overlap_param=2, use_argmax_for_center_detect=False)

    pipe = Pipeline(ImageLog('original', 'output_images')
        | Undistort('undistort', calib['camera_matrix'], calib['distortion_coefs'], calib['image_size']) | ImageLog('undistorted', 'output_images')
        | [
            ColorThreshold('color_threshold', 80, 200) | ImageLog('color_threshold_log', 'output_images'),
            GradientThreshold('gradient_threshold', [50, 250], [0.3, 0.9]) | ImageLog('gradient_threshold_log', 'output_images')]
        | MaxMerger('max_merger') | ImageLog('max_merger_out', 'output_images')
        | forward_warp | ImageLog('perspective_warp_out', 'output_images')
        | rough_lane_detector | PolyImageLog('poly_log', 'output_images')
        | MaskWithPoly('mask_with_poly') | ImageLog('mask_with_poly_out', 'output_images')
        | finer_lane_detector | PolyImageLog('poly_refined_log', 'output_images')
        | backward_warp | ImageLog('perspective_back_warp_out', 'output_images')
    )

    print(pipe)

    images = collect_images(input_dir)
    pipe.consume(tqdm(image_loader(images), total=len(images)))

if __name__ == "__main__":
    main()