import airsim
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, wait
from collections import defaultdict
import threading
from typing import Dict, List, Tuple


class DroneDetector:
    def __init__(self, client):
        self.client = client
        self.camera_name = "high_res"
        self.image_type = airsim.ImageType.Scene
        self.lock = threading.Lock()  # Add thread safety

    def setup_detection_filters(self, drone: str) -> None:
        """Set up detection filters once per drone"""
        with self.lock:  # Ensure thread-safe API calls
            self.client.simSetDetectionFilterRadius(
                self.camera_name,
                self.image_type,
                200 * 100,
                vehicle_name=drone
            )
            self.client.simAddDetectionFilterMeshName(
                self.camera_name,
                self.image_type,
                "BP_CrowdCharacter*",
                vehicle_name=drone
            )

    def get_geo_points(self, drone: str) -> Dict[int, List]:
        """Get detections for a specific drone"""
        with self.lock:  # Ensure thread-safe API calls
            chars = self.client.simGetDetections(
                self.camera_name,
                self.image_type,
                vehicle_name=drone
            )

        return {
            int(char.name.split('_')[-1]): [char.geo_point, char.name]
            for char in chars
        }


def process_drone(detector: DroneDetector,
                  drone: str,
                  pedestrian_data: List[str]) -> Tuple[str, List[Tuple]]:
    """Process a single drone's detections"""
    print(f"Starting processing for {drone}")
    los_positions = []

    with detector.lock:  # Thread-safe home point retrieval
        home = detector.client.getHomeGeoPoint(vehicle_name=drone)

    # Set up filters once per drone
    detector.setup_detection_filters(drone)

    # Get all geo points for this drone at once
    geo_points = detector.get_geo_points(drone)

    # Process all pedestrians
    for line in pedestrian_data:
        if not line.strip():
            continue

        ped_id, pos_id, x, y, z = map(float, line.split())
        ped_id = int(ped_id)

        if ped_id in geo_points:
            target = geo_points[ped_id][0]
            with detector.lock:  # Thread-safe line of sight check
                has_los = detector.client.simTestLineOfSightBetweenPoints(home, target)

            if has_los:
                los_positions.append((ped_id, pos_id, x, y, z))

    print(f"Completed processing for {drone}")
    return drone, los_positions


def save_pedestrian_los(client: airsim.MultirotorClient,
                        drone_names: List[str],
                        timestep: int) -> None:
    """Main function with synchronized parallel processing"""
    detector = DroneDetector(client)

    # Read pedestrian data once
    with open(f'matchings/Pedestrians/3d_{timestep:04d}.txt', 'r') as f:
        pedestrian_data = f.readlines()

    # Create output directory
    os.makedirs('matchings/Pedestrians/LoS', exist_ok=True)

    # Process all drones concurrently with exactly 8 workers
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks and get future objects
        futures = [
            executor.submit(
                process_drone,
                detector,
                drone,
                pedestrian_data
            )
            for drone in drone_names
        ]

        # Wait for all futures to complete
        print("Waiting for all drones to complete processing...")
        wait(futures)

        # Process results only after all drones are complete
        for future in futures:
            try:
                drone, los_positions = future.result()
                output_file = f'matchings/Pedestrians/LoS/{drone}_3d_{timestep:04d}.txt'

                with open(output_file, 'w') as f:
                    for pos in los_positions:
                        f.write(f"{pos[0]:.0f} {pos[1]:.0f} {pos[2]:.2f} {pos[3]:.2f} {pos[4]:.2f}\n")

                print(f"Saved LoS pedestrian positions for {drone} at timestep {timestep}")
            except Exception as e:
                print(f"Error processing {drone}: {str(e)}")

    print("All drones have completed processing")


if __name__ == '__main__':
    # Connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()

    # List of drone names
    drone_names = [f"Drone{i}" for i in range(1, 9)]  # Drone1 to Drone8

    # Example usage
    timestep = 0  # Replace with actual timestep
    save_pedestrian_los(client, drone_names, timestep)
    print("Proceeding with next steps...")