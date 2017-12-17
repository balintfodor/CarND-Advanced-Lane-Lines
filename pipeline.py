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
        im_id, image = item
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
        im_id, image = item
        und = cv2.remap(image, self.mapx, self.mapy, cv2.INTER_LINEAR)
        self.push((im_id, und))


class ColorThreshold(Node):
    def __init__(self, name, sat_th, val_th):
        super().__init__(name)
        self.sat_th = sat_th
        self.val_th = val_th

    def process(self, item):
        im_id, image = item
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        bin_image = (hsv[:, :, 1] >= self.sat_th) * (hsv[:, :, 2] >= self.val_th)
        out = (bin_image * 255).astype(np.uint8)
        self.push((im_id, out))


class GradientThreshold(Node):
    def __init__(self, name, mag_limits, dir_limits, kernel_size=5):
        super().__init__(name)
        self.kernel_size = kernel_size
        self.mag_limits = mag_limits
        self.dir_limits = [
            np.maximum(dir_limits[0], 0) * np.pi/2, 
            np.minimum(dir_limits[1], 1) * np.pi/2]

    def process(self, item):
        im_id, image = item
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
        self.push((im_id, out))


class MaxMerger(GroupByNode):
    def key(self, item):
        return item[0]

    def process(self, batch):
        images = [im for _, im in batch]
        out = np.max(np.array(images), axis=0)
        self.push((batch[0][0], out))


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
        im_id, image = item
        if self.inverse:
            out = cv2.warpPerspective(image, self.matrix, self.src_size, flags=cv2.WARP_INVERSE_MAP)
        else:
            out = cv2.warpPerspective(image, self.matrix, self.dst_size)
        self.push((im_id, out))

class LaneDetector(Node):
    def __init__(self, name, win_height=32):
        super().__init__(name)
        self.win_height = win_height

    def analyze_win(self, im):
        vert_sum = np.sum(im, axis=0)
        x = np.argmax(vert_sum)
        return x

    def robust_fit_poly(self, points):
        points = np.array(points)

        def f(p, x, y):
            return p[0] * x*x + p[1] * x + p[2] - y
        
        p0 = np.array([1, 1, 1])
        res = least_squares(f, p0, loss='soft_l1', f_scale=0.1,
            args=(points[:, 1], points[:, 0]))
        return res.x

    def draw_poly(self, im, p, xs):
        x = xs[0]
        y = p[0] * x*x + p[1] * x + p[2]
        for x_new in xs[1:]:
            y_new = p[0] * x*x + p[1] * x + p[2]
            cv2.line(im, (int(y), int(x)), (int(y_new), int(x_new)), (255,))
            y = y_new
            x = x_new

    def process(self, item):
        im_id, image = item
        wh = self.win_height
        half = image.shape[1] // 2
        overlap_param = 4
        step = wh // overlap_param

        out = np.copy(image) // 2

        h_samples = range(255 - wh, 0, -step)

        for side_offset in [0, half]:
            points = []
            for y_pos in h_samples:
                roi = image[y_pos:(y_pos + wh), side_offset:(side_offset + half)]
                x = self.analyze_win(roi) + side_offset
                y = y_pos + wh / 2
                points.append((x, y))
                cv2.circle(out, (int(x), int(y)), 2, (255,))
            poly = self.robust_fit_poly(points)
            self.draw_poly(out, poly, np.array(h_samples) + wh / 2)
        self.push((im_id, out))


def collect_images(folder):
    images = []
    for format in ['jpg', 'png', 'tif']:
        images.extend(glob.glob("{}/*.{}".format(folder, format)))
    return images

def image_loader(images):
    for i, im_path in enumerate(images):
        im = cv2.imread(im_path)
        yield (i, im)

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

    pipe = Pipeline(ImageLog('original', 'output_images')
        | Undistort('undistort', calib['camera_matrix'], calib['distortion_coefs'], calib['image_size']) | ImageLog('undistorted', 'output_images')
        | [
            ColorThreshold('color_threshold', 80, 200) | ImageLog('color_threshold_log', 'output_images'),
            GradientThreshold('gradient_threshold', [50, 250], [0.3, 0.9]) | ImageLog('gradient_threshold_log', 'output_images')]
        | MaxMerger('max_merger') | ImageLog('max_merger_out', 'output_images')
        | forward_warp | ImageLog('perspective_warp_out', 'output_images')
        | LaneDetector('lane_detector_rough') | ImageLog('lane_detector_out', 'output_images')
        | backward_warp | ImageLog('perspective_back_warp_out', 'output_images')
    )

    print(pipe)

    images = collect_images(input_dir)
    pipe.consume(tqdm(image_loader(images), total=len(images)))

if __name__ == "__main__":
    main()