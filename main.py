import airsim
import cv2
import numpy as np
import time
import random
import os
from generateMatchings import get_3d_points, generate_matchings
from getPedestrian import get_pedestrian
from generatePOM import generate_POM
from generateAnnotation import annotate
from pedestrianLoS import save_pedestrian_los
from datasetParameters import *

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

class CameraTracker:
    def __init__(self, client, drone_names, interval, max_timesteps, visualise):
        self.visualise = visualise
        self.client = client
        self.drone_names = drone_names
        self.camera_name = "high_res"
        self.image_type = airsim.ImageType.Scene
        self.image_width = 1920
        self.image_height = 1080
        self.snapshot_count = {drone: 0 for drone in drone_names}
        self.num_cams = NUM_CAM
        self.update_interval = interval # 0.5 seconds in simulation time
        self.max_timesteps = max_timesteps

        self.object_points = get_3d_points()

        # PID controllers for yaw and pitch (one set per drone)
        self.yaw_pid = {drone: PIDController(kp=0.0002, ki=0.00002, kd=0.0001) for drone in drone_names}
        self.pitch_pid = {drone: PIDController(kp=0.0002, ki=0.00002, kd=0.0001) for drone in drone_names}

        # Set up detection filters for each drone
        for drone in drone_names:
            self.client.simSetDetectionFilterRadius(self.camera_name, self.image_type, 200 * 100, vehicle_name=drone)
            self.client.simAddDetectionFilterMeshName(self.camera_name, self.image_type, "Shape_Cone*", vehicle_name=drone)

        if self.visualise:
            cv2.namedWindow("Camera View with Object Tracking", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Camera View with Object Tracking", 1280, 720)

        # Define the designated airspace
        self.airspace = {
            'x': (-4, 4),
            'y': (-4, 4),
            'z': (-8, -7)
        }

        # Initialize the target position for each drone
        self.target_position = {drone: self.get_random_position() for drone in drone_names}

        self.home = client.getHomeGeoPoint(vehicle_name="Drone1")

    def get_random_position(self):
        return airsim.Vector3r(
            random.uniform(*self.airspace['x']),
            random.uniform(*self.airspace['y']),
            random.uniform(*self.airspace['z'])
        )

    def move_drone(self, drone):
        # Get current position
        current_position = self.client.getMultirotorState(vehicle_name=drone).kinematics_estimated.position

        # Calculate direction vector
        direction = self.target_position[drone] - current_position

        # Normalize direction vector
        distance = np.linalg.norm([direction.x_val, direction.y_val, direction.z_val])
        if distance > 0.1:  # If drone is not very close to the target
            normalized_direction = direction / distance
            velocity = normalized_direction * 2  # Adjust  multiplier to change speed

            # Move drone
            self.client.moveByVelocityAsync(velocity.x_val, velocity.y_val, velocity.z_val,
                                            self.update_interval, vehicle_name=drone)
        else:
            # If close to target, get a new random position
            self.target_position[drone] = self.get_random_position()

    def take_snapshot(self, drone, max_attempts=5, delay_between_attempts=1):
        for attempt in range(max_attempts):
            try:
                # Attempt to get the image
                responses = self.client.simGetImages(
                    [airsim.ImageRequest("high_res", airsim.ImageType.Scene, False, False)],
                    vehicle_name=drone)

                if not responses:
                    print(f"Attempt {attempt + 1}: No image received from AirSim for drone {drone}")
                    continue

                response = responses[0]

                # Create the directory if it doesn't exist
                directory = f"image_subsets/D{drone[-1]}"
                os.makedirs(directory, exist_ok=True)

                filename = f"{directory}/{self.snapshot_count[drone]:04d}"
                self.snapshot_count[drone] += 1

                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(response.height, response.width, 3)

                # Check if the image data is valid
                if img_rgb.size == 0 or img_rgb.shape[0] == 0 or img_rgb.shape[1] == 0:
                    print(f"Attempt {attempt + 1}: Invalid image data for drone {drone}")
                    continue

                # Try to save the image using airsim.write_png
                airsim.write_png(os.path.normpath(filename + '.png'), img_rgb)
                print(f"Saved snapshot: {filename}.png")
                return True

            except Exception as e:
                print(f"Attempt {attempt + 1}: Error capturing/saving image for drone {drone}: {str(e)}")

            # Wait before the next attempt
            time.sleep(delay_between_attempts)

        print(f"Failed to capture/save image for drone {drone} after {max_attempts} attempts")
        return False

    def get_image_and_detections(self, drone):
        rawImage = self.client.simGetImage(self.camera_name, self.image_type, vehicle_name=drone)
        png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
        detections = self.client.simGetDetections(self.camera_name, self.image_type, vehicle_name=drone)
        return png, detections

    def get_object_center(self, detections):
        if detections:
            char = detections[0]
            center_x = int((char.box2D.min.x_val + char.box2D.max.x_val) / 2)
            center_y = int((char.box2D.min.y_val + char.box2D.max.y_val) / 2)
            return (center_x, center_y), char.box2D
        return None, None

    def adjust_camera(self, drone, object_center):
        if object_center is None:
            return

        center_x, center_y = object_center
        frame_center_x = self.image_width / 2
        frame_center_y = self.image_height / 2

        # Calculate errors
        error_x = center_x - frame_center_x
        error_y = center_y - frame_center_y

        # Compute PID outputs
        yaw_adjustment = self.yaw_pid[drone].compute(error_x, self.update_interval)
        pitch_adjustment = -self.pitch_pid[drone].compute(error_y, self.update_interval)  # Negative because pitch is inverted

        # Get current poses
        camera_pose = self.client.simGetCameraInfo(self.camera_name, vehicle_name=drone).pose
        drone_pose = self.client.simGetVehiclePose(vehicle_name=drone)

        # Extract current orientations
        current_pitch, _, current_yaw_cam = airsim.to_eularian_angles(camera_pose.orientation)
        _, _, current_yaw = airsim.to_eularian_angles(drone_pose.orientation)

        # Apply adjustments
        new_yaw = current_yaw + yaw_adjustment
        new_pitch = current_pitch + pitch_adjustment

        # Set new drone orientation
        new_drone_pose = airsim.Pose(drone_pose.position, airsim.to_quaternion(0, 0, new_yaw))
        self.client.simSetVehiclePose(new_drone_pose, False, drone)

        # Set new camera orientation
        new_camera_pose = airsim.Pose(airsim.Vector3r(0.5, 0, 0.1), airsim.to_quaternion(new_pitch, 0, 0))
        self.client.simSetCameraPose(self.camera_name, new_camera_pose, vehicle_name=drone)

    def update_display(self, image, center, box, drone):
        if box:
            start_point = (int(box.min.x_val), int(box.min.y_val))
            end_point = (int(box.max.x_val), int(box.max.y_val))
            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)

        if center:
            cv2.circle(image, center, 5, (0, 0, 255), -1)
            diff_x = center[0] - self.image_width / 2
            diff_y = center[1] - self.image_height / 2
            cv2.putText(image, f"{drone} X diff: {diff_x:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"{drone} Y diff: {diff_y:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Camera View with Object Tracking", image)
        cv2.waitKey(1)

    def track_objects(self):
        timestep = 0
        while True:
            current_time = time.time()
            for drone in self.drone_names:
                # Move the drone
                self.move_drone(drone)

            # Continue simulation for the specified interval
            new_time = time.time() - current_time
            pause_time = self.update_interval - new_time
            self.client.simContinueForTime(pause_time)

            # Update pedestrian and generate matchings
            self.client.simPause(False)
            get_pedestrian(self.client, self.camera_name, self.drone_names, timestep)
            self.client.simPause(True)

            for drone in self.drone_names:
                # Get image and detections
                image, detections = self.get_image_and_detections(drone)
                object_center, box = self.get_object_center(detections)
                self.adjust_camera(drone, object_center)

                if self.visualise:
                    self.update_display(image, object_center, box, drone)

            save_pedestrian_los(self.client, self.drone_names, timestep)
            generate_matchings(self.client, self.camera_name, self.drone_names, self.object_points, timestep)
            for droneSnap in self.drone_names:
                self.take_snapshot(droneSnap)

            timestep += 1

            if timestep >= self.max_timesteps:
                break

            self.client.simPause(False)


if __name__ == '__main__':
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.simPause(False)

    # List of drone names
    drone_names = [f"Drone{i}" for i in range(1, NUM_CAM + 1)]  # Drone1 to Drone8

    interval = 0.5
    max_timesteps = 10
    visualise = False

    # Define waypoints [(x, y, z), ...]
    waypoints = [
        (-3, -3, -8),
        (0, -3, -8),
        (3, -3, -8),
        (3, 0, -8),
        (3, 3, -8),
        (0, 3, -8),
        (-3, 3, -8),
        (-3, 0, -8)
    ]

    # Enable API control and take off for all drones
    for i, drone in enumerate(drone_names):
        client.enableApiControl(True, drone)
        client.armDisarm(True, drone)
        client.takeoffAsync(vehicle_name=drone).join()

        # Move to assigned waypoint
        # x, y, z = waypoints[i]
        # client.moveToPositionAsync(x, y, z, 1, vehicle_name=drone).join()
        client.moveToPositionAsync(0, 0, -7, 1, vehicle_name=drone).join()

    tracker = CameraTracker(client, drone_names, interval, max_timesteps, visualise)
    tracker.track_objects()

    # Land all drones after the tracking is done
    for drone in drone_names:
        client.simPause(False)
        client.landAsync(vehicle_name=drone)
        client.armDisarm(False, drone)
        client.enableApiControl(False, drone)

    for timestep in range(max_timesteps):
        generate_POM(timestep)
        annotate(timestep, max_timesteps)