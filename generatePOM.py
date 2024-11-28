import os
from datasetParameters import *
from unitConversion import *
from camera_visualizer import *

def generate_cam_pom(rvec, tvec, cameraMatrix, distCoeffs):
    # WILDTRACK has irregular denotion: H*W=480*1440, normally x would be \in [0,1440), not [0,480)
    # In our data annotation, we follow the regular x \in [0,W), and one can calculate x = pos % W, y = pos // W
    coord_x, coord_y = get_worldcoord_from_pos(np.arange(MAP_HEIGHT * MAP_WIDTH * MAP_EXPAND * MAP_EXPAND))
    centers3d = np.stack([coord_x, coord_y, np.zeros_like(coord_y)], axis=1)
    points3d8s = []
    points3d8s.append(centers3d + np.array([MAN_RADIUS, MAN_RADIUS, 0]))
    points3d8s.append(centers3d + np.array([-MAN_RADIUS, MAN_RADIUS, 0]))
    points3d8s.append(centers3d + np.array([MAN_RADIUS, -MAN_RADIUS, 0]))
    points3d8s.append(centers3d + np.array([-MAN_RADIUS, -MAN_RADIUS, 0]))
    points3d8s.append(centers3d + np.array([MAN_RADIUS, MAN_RADIUS, MAN_HEIGHT]))
    points3d8s.append(centers3d + np.array([-MAN_RADIUS, MAN_RADIUS, MAN_HEIGHT]))
    points3d8s.append(centers3d + np.array([MAN_RADIUS, -MAN_RADIUS, MAN_HEIGHT]))
    points3d8s.append(centers3d + np.array([-MAN_RADIUS, -MAN_RADIUS, MAN_HEIGHT]))
    bbox = np.ones([centers3d.shape[0], 4]) * np.array([IMAGE_WIDTH, IMAGE_HEIGHT, 0, 0])  # xmin,ymin,xmax,ymax
    for i in range(8):  # for all 8 points
        points_img, _ = cv2.projectPoints(points3d8s[i], rvec, tvec, cameraMatrix, distCoeffs)
        points_img = points_img.squeeze()
        bbox[:, 0] = np.min([bbox[:, 0], points_img[:, 0]], axis=0)  # xmin
        bbox[:, 1] = np.min([bbox[:, 1], points_img[:, 1]], axis=0)  # ymin
        bbox[:, 2] = np.max([bbox[:, 2], points_img[:, 0]], axis=0)  # xmax
        bbox[:, 3] = np.max([bbox[:, 3], points_img[:, 1]], axis=0)  # xmax
        pass
    points_img, _ = cv2.projectPoints(centers3d, rvec, tvec, cameraMatrix, distCoeffs)
    points_img = points_img.squeeze()
    bbox[:, 3] = points_img[:, 1]
    # offset = points_img[:, 0] - (bbox[:, 0] + bbox[:, 2]) / 2
    # bbox[:, 0] += offset
    # bbox[:, 2] += offset
    notvisible = np.zeros([centers3d.shape[0]])
    notvisible += (bbox[:, 0] >= IMAGE_WIDTH - 2) + (bbox[:, 1] >= IMAGE_HEIGHT - 2) + \
                  (bbox[:, 2] <= 1) + (bbox[:, 3] <= 1)
    # notvisible += bbox[:, 2] - bbox[:, 0] > bbox[:, 3] - bbox[:, 1]  # w > h
    notvisible += (bbox[:, 2] - bbox[:, 0] > IMAGE_WIDTH / 3)  # + (bbox[:, 3] - bbox[:, 1] > IMAGE_HEIGHT / 3)
    return bbox.astype(int), notvisible.astype(bool)


def generate_POM(timestep):
    fpath = f'POMs/rectangles_{timestep:04d}.pom'
    if os.path.exists(fpath):
        os.remove(fpath)
    fp = open(fpath, 'w')

    for cam in range(NUM_CAM):
        intrinsic_file = f'calibrations/intrinsic/intr_Drone{cam + 1}_{timestep:04d}.xml'
        extrinsic_file = f'calibrations/extrinsic/extr_Drone{cam + 1}_{timestep:04d}.xml'

        fp_calibration = cv2.FileStorage(intrinsic_file, flags=cv2.FILE_STORAGE_READ)
        cameraMatrix = fp_calibration.getNode('camera_matrix').mat()
        distCoeffs = fp_calibration.getNode('distortion_coefficients').mat()
        fp_calibration.release()

        fp_calibration = cv2.FileStorage(extrinsic_file, flags=cv2.FILE_STORAGE_READ)
        rvec = fp_calibration.getNode('rvec').mat().squeeze()
        tvec = fp_calibration.getNode('tvec').mat().squeeze()
        fp_calibration.release()

        # visualize_camera(cameraMatrix, rvec, tvec, MAP_WIDTH, MAP_HEIGHT)

        bbox, notvisible = generate_cam_pom(rvec, tvec, cameraMatrix, distCoeffs)  # xmin,ymin,xmax,ymax

        for pos in range(len(notvisible)):
            if notvisible[pos]:
                fp.write(f'RECTANGLE {cam} {pos} notvisible\n')
            else:
                fp.write(f'RECTANGLE {cam} {pos} '
                         f'{bbox[pos, 0]} {bbox[pos, 1]} {bbox[pos, 2]} {bbox[pos, 3]}\n')

    fp.close()
    print(f"Generated POM for timestep {timestep}")


if __name__ == '__main__':
    for timestep in range(1):  # Assuming 10 timesteps from 0000 to 0009
        generate_POM(timestep)