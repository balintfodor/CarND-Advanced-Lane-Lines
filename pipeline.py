import cv2
import numpy as np
from consecution import Pipeline, Node, GroupByNode
import glob
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


class SaturationThreshold(Node):
    def __init__(self, name, threshold):
        super().__init__(name)
        self.th = threshold

    def process(self, item):
        im_id, image = item
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        bin_image = hsv[:, :, 1] >= self.th
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
    input_dir = 'test_images'

    calib = np.load(calib_file)

    pipe = Pipeline(ImageLog('original', 'output_images')
        | Undistort('undistort', calib['camera_matrix'], calib['distortion_coefs'], calib['image_size']) | ImageLog('undistorted', 'output_images')
        | [
            SaturationThreshold('saturation_threshold', 150) | ImageLog('saturation_threshold_log', 'output_images'),
            GradientThreshold('gradient_threshold', [50, 250], [0.3, 0.9]) | ImageLog('gradient_threshold_log', 'output_images')]
        | MaxMerger('max_merger') | ImageLog('max_merger_out', 'output_images')
    )

    print(pipe)

    images = collect_images(input_dir)
    pipe.consume(tqdm(image_loader(images), total=len(images)))

if __name__ == "__main__":
    main()