import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import xml.etree.ElementTree as ET
import base64
import struct
from unitConversion import get_worldcoord_from_worldgrid


def parse_xml_matrix(file_path, matrix_name):
    tree = ET.parse(file_path)
    root = tree.getroot()
    matrix = root.find(matrix_name)
    rows = int(matrix.find('rows').text)
    cols = int(matrix.find('cols').text)
    dt = matrix.find('dt').text
    data = matrix.find('data')

    if data.get('type_id') == 'binary':
        binary_data = base64.b64decode(data.text)
        if dt == 'd':
            fmt = 'd'
            size = 8
        elif dt == 'f':
            fmt = 'f'
            size = 4
        else:
            raise ValueError(f"Unsupported data type: {dt}")

        try:
            values = struct.unpack(f'{rows * cols}{fmt}', binary_data[:rows * cols * size])
        except struct.error:
            all_values = struct.unpack(f'{len(binary_data) // size}{fmt}', binary_data)
            values = all_values[:rows * cols]
    else:
        values = list(map(float, data.text.split()))

    return np.array(values, dtype=float).reshape((rows, cols))


def parse_intrinsics(file_path):
    fp_calibration = cv2.FileStorage(file_path, flags=cv2.FILE_STORAGE_READ)
    camera_matrix = fp_calibration.getNode('camera_matrix').mat()
    distortion_coefficients = fp_calibration.getNode('distortion_coefficients').mat()
    fp_calibration.release()
    return camera_matrix, distortion_coefficients


def parse_extrinsics(file_path):
    fp_calibration = cv2.FileStorage(file_path, flags=cv2.FILE_STORAGE_READ)
    rvec = fp_calibration.getNode('rvec').mat().squeeze()
    tvec = fp_calibration.getNode('tvec').mat().squeeze()
    fp_calibration.release()
    return rvec, tvec


def project_points(world_coord, rvec, tvec, camera_matrix, dist_coeffs):
    # Project 3D points to 2D image plane
    image_points, _ = cv2.projectPoints(world_coord, rvec, tvec, camera_matrix, dist_coeffs)
    return image_points.squeeze()


if __name__ == '__main__':
    time = 10
    img = Image.open(f'image_subsets/D1/{time:04d}.png')
    xi = np.arange(0, 600, 30)
    yi = np.arange(0, 600, 30)
    world_grid = np.stack(np.meshgrid(xi, yi, indexing='ij')).reshape([2, -1])
    world_coord = get_worldcoord_from_worldgrid(world_grid)

    # Add Z-coordinate (-5) to world coordinates
    world_coord_3d = np.vstack((world_coord, np.full((1, world_coord.shape[1]), -5)))

    print("World coordinates shape:", world_coord_3d.shape)
    print("Sample world coordinates:\n", world_coord_3d[:, :5])

    # Parse intrinsic and extrinsic matrices
    camera_matrix, dist_coeffs = parse_intrinsics(f"calibrations/intrinsic/intr_Drone1_{time:04d}.xml")
    rvec, tvec = parse_extrinsics(f"calibrations/extrinsic/extr_Drone1_{time:04d}.xml")

    # Project world coordinates to image coordinates
    img_coord = project_points(world_coord_3d.T, rvec, tvec, camera_matrix, dist_coeffs)

    # Filter points outside the image
    valid_points = (img_coord[:, 0] >= 0) & (img_coord[:, 1] >= 0) & \
                   (img_coord[:, 0] < 1920) & (img_coord[:, 1] < 1080)
    img_coord = img_coord[valid_points]

    # Visualize the grid
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.scatter(img_coord[:, 0], img_coord[:, 1], c='r', s=10)
    plt.title("Grid Visualisation")
    plt.savefig('grid_visualisation.png')
    plt.show()