import argparse
import glob, os
import cv2
import numpy as np
from tqdm import tqdm

def parse_args():
    '''arg parsing'''
    parser = argparse.ArgumentParser(description='Camera Calibration')
    parser.add_argument('input_folder', type=str, help='a folder containing chessboard images')
    parser.add_argument('output_folder', type=str, help='output folder')
    return parser.parse_args()

def collect_images(folder):
    images = []
    for format in ['jpg', 'png', 'tif']:
        images.extend(glob.glob("{}/*.{}".format(folder, format)))
    return images

def collect_corners(images, board_size, args):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

    obj_points = []
    im_points = []
    im_path_processed = []
    im_size = (0, 0)

    print('chessboard detection')
    for i, im_path in tqdm(enumerate(images), total=len(images)):
        im = cv2.imread(im_path)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_size = (gray.shape[1], gray.shape[0])
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if ret:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1,-1), criteria)

            obj_points.append(objp)
            im_points.append(corners_refined)
            im_path_processed.append(im_path)

            im = cv2.drawChessboardCorners(im, board_size, corners_refined, ret)
            cv2.imwrite('{}/{:02d}.png'.format(args.output_folder, i), im)

    return {'obj_points': obj_points, 'im_points': im_points,
        'im_path_processed': im_path_processed, 'im_size': im_size,
        'board_size': board_size}

def calibrate(detected):
    print('calibration')
    repro_error, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(detected['obj_points'], 
        detected['im_points'], detected['im_size'], None, None)
    return {'camera_matrix': camera_matrix, 'distortion': distortion,
        'rvecs': rvecs, 'tvecs': tvecs}, repro_error

def reproject(detected, calib_data, args):
    print('reprojection')
    images = detected['im_path_processed']
    for i, im_path in tqdm(enumerate(images), total=len(images)):
        pts, _ = cv2.projectPoints(detected['obj_points'][i], calib_data['rvecs'][i], calib_data['tvecs'][i],
            calib_data['camera_matrix'], calib_data['distortion'])
        im = cv2.imread(im_path)
        im = cv2.drawChessboardCorners(im, detected['board_size'], pts, True)
        cv2.imwrite('{}/repro-{:02d}.png'.format(args.output_folder, i), im)

def undistort(detected, calib_data, args):
    print('undistortion')
    mapx, mapy = cv2.initUndistortRectifyMap(calib_data['camera_matrix'],
        calib_data['distortion'], None, None, detected['im_size'], calib_data['distortion'].size)
    images = detected['im_path_processed']
    for i, im_path in tqdm(enumerate(images), total=len(images)):
        im = cv2.imread(im_path)
        und = cv2.remap(im, mapx, mapy, cv2.INTER_LINEAR)
        cv2.imwrite('{}/undist-{:02d}.png'.format(args.output_folder, i), und)

def main():
    args = parse_args()
    chessboard_size = (9, 6)
    images = collect_images(args.input_folder)
    detected = collect_corners(images, chessboard_size, args)
    calib_data, repro_error = calibrate(detected)

    reproject(detected, calib_data, args)
    undistort(detected, calib_data, args)

    print('camera matrix\n{}'.format(calib_data['camera_matrix']))
    print('distortion\n{}'.format(calib_data['distortion']))

    n = len(detected['im_path_processed'])
    print('reprojection error for {} images with size {} with {} chessboard size is {}'.format(
        n, detected['im_size'], chessboard_size, repro_error))

    calib_out_file = 'calibration.npz'
    np.savez('{}/{}'.format(args.output_folder, calib_out_file),
        camera_matrix=calib_data['camera_matrix'],
        distortion_coefs=calib_data['distortion'])
    print('calibration saved to {}/{}'.format(args.output_folder, calib_out_file))

if __name__ == "__main__":
    main()
