import cv2
import numpy as np
import airsim
import socket
import time
import math
import os

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


def get_3d_points(client):
    # Dictionary to store the 3D points
    points_3d = {}

    # Pattern to search for your static mesh actors
    plane_pattern = "StaticMeshActor_UAID_4CEDF"

    # Get all scene objects
    max_attempts = 5
    attempt = 0

    while len(points_3d) < 300 and attempt < max_attempts:
        attempt += 1
        print(f"Attempt {attempt} to get scene objects")

        try:
            # Get all objects in the scene
            object_names = client.simListSceneObjects()

            # Filter objects based on pattern
            plane_objects = [name for name in object_names if plane_pattern in name]

            # Get pose for each matching object
            for obj_name in plane_objects:
                # Check if we already have this object
                if obj_name in points_3d:
                    continue

                # Get pose of the object
                pose = client.simGetObjectPose(obj_name)

                # Extract coordinates
                x, y, z = pose.position.x_val*100, pose.position.y_val*100, pose.position.z_val*-100

                if int(z) == -490:
                    points_3d[obj_name] = [x, y, z]
                    print(f"Collected point for {obj_name}: X={x}, Y={y}, Z={z}")

        except Exception as e:
            print(f"An error occurred while collecting 3D points: {e}")
            time.sleep(2)

    # Process 3D points
    object_points = []
    for obj_name, coords in points_3d.items():
        # Extract plane number from object name
        try:
            plane_number = int(obj_name.split('_')[-1])
            object_points.append([0, plane_number] + coords)
        except (ValueError, IndexError) as e:
            print(f"Could not extract plane number from {obj_name}: {e}")

    # Save points to files
    if object_points:
        np.savetxt(f'matchings/Planes/3d.txt', object_points, fmt='%d %d %.6f %.6f %.6f')
        print(f"Saved {len(object_points)} points to matchings/Planes/3d.txt")
    else:
        print("No points to save")

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
        except Exception as e:
            print(f"Unexpected error during calibration for {drone}: {e}")
            continue


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

    return camera_matrices, dist_coeffs, rvecs_dict, tvecs_dict


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

    camera_matrices, dist_coeffs, rvecs, tvecs = calibrate_cameras(client, image_points, object_points, camera_name, timestep)

    save_2d_3d_points(image_points, object_points, drone_names, timestep)


if __name__ == '__main__':
    client = airsim.MultirotorClient()
    client.confirmConnection()
    camera_name = "high_res"
    drone_names = [f"Drone{i}" for i in range(1, 9)]  # Drone1 to Drone8

    object_points = get_3d_points()
    generate_matchings(client, camera_name, drone_names, object_points)
