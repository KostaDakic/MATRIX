import airsim
import os
import time
import socket
import select
from unitConversion import *

class UnrealClient:
    def __init__(self, host='localhost', port=8000):
        self.host = host
        self.port = port
        self.socket = None
        self.connect()

    def connect(self):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
                print(f"Connected to {self.host}:{self.port}")
                return
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise

    def reconnect(self):
        print("Attempting to reconnect...")
        self.close()
        self.connect()

    def close(self):
        if self.socket:
            self.socket.close()

def get_drone_data(s, timeout=1):
    ped_data = {}
    buffer = ""
    start_time = time.time()
    while len(ped_data) < 100 and time.time() - start_time < timeout:
        try:
            ready, _, _ = select.select([s], [], [], timeout)
            if ready:
                data = s.recv(4096).decode('utf-8')  # Increased buffer size
                if not data:
                    print("No data received, connection might be closed")
                    return None
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    parts = line.strip().split(',')
                    if parts[0].startswith('BP_CrowdCharacter'):
                        name, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
                        # ped_data[name] = [x, y, z]
                        if 0 < x < 3000 and 0 < y < 3000:  # Worldgrid
                            ped_data[name] = [x, y, z]
            else:
                print("Timeout waiting for drone data")
                return None
        except Exception as e:
            print(f"Error receiving data: {e}")
            return None
    return ped_data

def generate_matchings(ped_data, cam):
    points_3d = []
    for data in ped_data.items():
        pos_id = get_pos_from_worldcoord([float(data[1][0])/100, float(data[1][1])/100])
        coords = [float(data[1][0])/100, float(data[1][1])/100, 0]
        if len(data[0]) == 21:
            ped_id = int(data[0][-1])
        elif len(data[0]) == 22:
            ped_id = int(data[0][-2:])
        else:
            ped_id = int(data[0][-3:])
        points_3d.append([ped_id, pos_id] + coords)
    return points_3d

def capture_images(drone_names, camera_name, timestep):
    for drone in drone_names:
        try:
            # Save image
            responses = client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)],
                                            vehicle_name=drone)
            response = responses[0]
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            airsim.write_png(os.path.normpath(f"image_subsets/Test/D{drone[-1]}/image_{timestep:04d}.png"), img_rgb)
        except Exception as e:
            print(f"Error processing drone {drone}: {e}")

def get_pedestrian(client, camera_name, drone_names, timestep):
    # Unreal Engine server details
    UE_HOST = '127.0.0.1'
    UE_PORT = 8000

    ped_matchings = {drone: [] for drone in drone_names}

    ue_client = UnrealClient(UE_HOST, UE_PORT)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((UE_HOST, UE_PORT))
        s.setblocking(0)  # Set socket to non-blocking mode
        try:
            ped_data = get_drone_data(s)
            if ped_data is None:
                print(f"Failed to get drone data. Attempting to reconnect...")
                ue_client.reconnect()
                s.close()
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((UE_HOST, UE_PORT))
                s.setblocking(0)


            # client.simPause(True)  # Pause the simulation
            # capture_images(drone_names, camera_name, timestep)

            # Process the results
            for cam, drone in enumerate(drone_names):
                ped_matchings[drone].append(generate_matchings(ped_data, cam))


            # Resume the simulation
            # client.simPause(False)

        except Exception as e:
            print(f"An error occurred while collecting pedestrian points: {e}")

        # Clear any leftover data in the socket
        while True:
            ready, _, _ = select.select([s], [], [], 0)
            if not ready:
                break
            _ = s.recv(4096)  # Increased buffer size

    ue_client.close()

    # Save the matchings
    with open(f'matchings/Pedestrians/3d_{timestep:04d}.txt', 'w') as f:  # Changed 'w' to 'a'
        for timestep_data in ped_matchings[drone_names[0]]:
            for detection in timestep_data:
                f.write(f"{detection[0]} {detection[1]} {detection[2]:.6f} {detection[3]:.6f} {detection[4]:.6f}\n")
            f.write("\n")
    print("3D Pedestrian points saved.")


if __name__ == '__main__':
    # Connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()

    # Set camera name and image type
    camera_name = "high_res"
    image_type = airsim.ImageType.Scene

    # List of drone names
    drone_names = [f"Drone{i}" for i in range(1, 9)]  # Drone1 to Drone8

    total_duration = 10  # seconds
    captures_per_second = 2
    timestep = 0000
    get_pedestrian(client, camera_name, drone_names, timestep)