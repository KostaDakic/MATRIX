import cv2
import numpy as np
import airsim
import socket
import time
import math
import os
import shutil

def get_2d_points(client, drone_names, camera_name, image_type):
    all_detections = {}
    for drone in drone_names:
        client.simSetDetectionFilterRadius(camera_name, image_type, 200 * 100, vehicle_name=drone)
        client.simAddDetectionFilterMeshName(camera_name, image_type, "Shape_Plane*", vehicle_name=drone)
        rawImage = client.simGetImage(camera_name, image_type, vehicle_name=drone)
        if not rawImage:
            continue
        chars = client.simGetDetections(camera_name, image_type, vehicle_name=drone)

        # Create a dictionary for this drone's detections
        drone_detections = {}
        for char in chars:
            object_name = char.name
            x = char.box2D.min.x_val
            y = char.box2D.min.y_val
            drone_detections[object_name] = [x, y]

        all_detections[drone] = drone_detections

    return all_detections


def get_3d_points():
    HOST, PORT = '127.0.0.1', 8000

    def connect_to_server():
        while True:
            try:
                print(f"Attempting to connect to {HOST}:{PORT}")
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((HOST, PORT))
                print(f"Connected to {HOST}:{PORT}")
                return s
            except ConnectionRefusedError:
                print(f"Connection to {HOST}:{PORT} was refused. Retrying in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                print(f"An error occurred: {e}. Retrying in 5 seconds...")
                time.sleep(5)

    points_3d = {}

    try:
        with connect_to_server() as s:
            buffer = ""
            while len(points_3d) < 100: # number of planes
                data = s.recv(1024).decode('utf-8')
                if not data:
                    print("No data received, connection might be closed")
                    break
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    parts = line.strip().split(',')
                    if len(parts) == 10 and parts[0].startswith('StaticMeshActor_UAID_4CEDF') and float(
                            parts[3]) == -490.0:
                        plane_name, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
                        if plane_name not in points_3d:
                            points_3d[plane_name] = [x, y, z]
                            print(f"Collected point for {plane_name}: X={x}, Y={y}, Z={z}")
    except Exception as e:
        print(f"An error occurred while collecting 3D points: {e}")

    # Process 3D points
    object_points = []
    for obj_name, coords in points_3d.items():
        plane_number = int(obj_name.split('_')[-1])
        object_points.append([0, plane_number] + coords)

    # Save points to files
    np.savetxt(f'matchings/Planes/3d.txt', object_points, fmt='%d %d %.6f %.6f %.6f')

    return points_3d


def get_camera_intrinsics(client, camera_name, drone_name):
    # camera_info = client.simGetCameraInfo(camera_name, vehicle_name=drone_name)
    request = airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
    response = client.simGetImages([request], vehicle_name=drone_name)[0]
    width = response.width
    height = response.height
    fov = 70
    focal_length = (width / 2) / math.tan(math.radians(fov / 2))

    return np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ])


def calibrate_cameras(client, image_points, object_points, camera_name, timestep):
    camera_matrices = {}
    dist_coeffs = {}
    rvecs_dict = {}
    tvecs_dict = {}

    # Create directories for saving calibration results
    os.makedirs('calibrations/intrinsic', exist_ok=True)
    os.makedirs('calibrations/extrinsic', exist_ok=True)

    for drone, drone_image_points in image_points.items():
        imgpoints = []
        objpoints = []

        for obj_name, point_2d in drone_image_points.items():
            if obj_name in object_points:
                imgpoints.append(point_2d)
                objpoints.append(object_points[obj_name])

        if len(imgpoints) < 4:
            print(f"Not enough points for calibration for {drone}. Skipping.")
            continue

        imgpoints = np.array(imgpoints, dtype=np.float32)
        imgpoints[:, 1] = imgpoints[:, 1]
        imgpoints[:, 0] = 1920 - imgpoints[:, 0]
        objpoints = np.array(objpoints, dtype=np.float32)
        objpoints = objpoints / 100.0  # Convert from cm to m
        objpoints[:, 2] += 5  # Add 5m to all Z coordinates

        # Get camera intrinsics
        cameraMatrix = get_camera_intrinsics(client, camera_name, drone)

        # Get image size
        request = airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
        response = client.simGetImages([request], vehicle_name=drone)[0]
        image_size = (response.width, response.height)

        try:
            _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                [objpoints],  # List of object point sets (only one set in this case)
                [imgpoints],  # List of image point sets (only one set in this case)
                image_size,
                cameraMatrix,
                None,
                flags=cv2.CALIB_USE_INTRINSIC_GUESS
            )

            camera_matrices[drone] = mtx
            dist_coeffs[drone] = dist
            rvecs_dict[drone] = rvecs[0]
            tvecs_dict[drone] = tvecs[0]

            # Save intrinsic parameters
            f = cv2.FileStorage(f'calibrations/intrinsic/intr_{drone}_{timestep:04d}.xml', flags=cv2.FILE_STORAGE_WRITE)
            f.write(name='camera_matrix', val=mtx)
            f.write(name='distortion_coefficients', val=dist)
            f.release()

            # Save extrinsic parameters
            f = cv2.FileStorage(f'calibrations/extrinsic/extr_{drone}_{timestep:04d}.xml', flags=cv2.FileStorage_WRITE_BASE64)
            f.write(name='rvec', val=rvecs[0])
            f.write(name='tvec', val=tvecs[0])
            f.release()

        except Exception as e:
            print(f"Unexpected error during calibration for {drone}: {e}")
            # Try to copy the previous timestep's calibration files
            prev_timestep = timestep - 1
            if prev_timestep >= 0:
                try:
                    # Copy intrinsic parameters from previous timestep
                    prev_intr_file = f'calibrations/intrinsic/intr_{drone}_{prev_timestep:04d}.xml'
                    curr_intr_file = f'calibrations/intrinsic/intr_{drone}_{timestep:04d}.xml'

                    if os.path.exists(prev_intr_file):
                        shutil.copy2(prev_intr_file, curr_intr_file)
                        print(f"Copied previous intrinsic calibration for {drone}")

                    # Copy extrinsic parameters from previous timestep
                    prev_extr_file = f'calibrations/extrinsic/extr_{drone}_{prev_timestep:04d}.xml'
                    curr_extr_file = f'calibrations/extrinsic/extr_{drone}_{timestep:04d}.xml'

                    if os.path.exists(prev_extr_file):
                        shutil.copy2(prev_extr_file, curr_extr_file)
                        print(f"Copied previous extrinsic calibration for {drone}")
                except Exception as copy_error:
                    print(f"Error copying previous calibration files for {drone}: {copy_error}")
            continue

def save_2d_3d_points(image_points, object_points, drone_names, timestep):
    os.makedirs('matchings', exist_ok=True)

    for cam, drone in enumerate(drone_names):
        # Process 2D points
        points_2d = []
        for obj_name, coords in image_points[drone].items():
            plane_number = int(obj_name.split('_')[-1])
            points_2d.append([0, plane_number, coords[0], coords[1]])

        # Process 3D points
        points_3d = []
        for obj_name, coords in object_points.items():
            plane_number = int(obj_name.split('_')[-1])
            points_3d.append([0, plane_number] + coords)

        # Save points to files
        np.savetxt(f'matchings/Planes/Camera{cam + 1}_2d_{timestep:04d}.txt', points_2d, fmt='%d %d %.6f %.6f')

    print(f"2D points saved for {len(drone_names)} cameras.")

def generate_matchings(client, camera_name, drone_names, object_points, timestep = 0):

    image_points = get_2d_points(client, drone_names, camera_name, airsim.ImageType.Scene)

    calibrate_cameras(client, image_points, object_points, camera_name, timestep)

    save_2d_3d_points(image_points, object_points, drone_names, timestep)


if __name__ == '__main__':
    client = airsim.MultirotorClient()
    client.confirmConnection()
    camera_name = "high_res"
    drone_names = [f"Drone{i}" for i in range(1, 9)]  # Drone1 to Drone8

    object_points = get_3d_points()
    generate_matchings(client, camera_name, drone_names, object_points)