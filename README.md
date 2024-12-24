# MATRIX (Multi-Aerial TRacking In compleX environments) Dataset Generator

## Overview

This project implements a multi-camera multiview system designed to generate detection and tracking datasets using eight drones in an Unreal Engine environment. The drones, controlled via the AirSim plugin by Microsoft, fly randomly and capture images, creating a rich dataset for computer vision tasks.

The dataset with 1000 timesteps, 40 pedestrians on a 30mx30m area can be downloaded [here](https://drive.google.com/file/d/1hSB72MSPQLEIL-9Hb0DoBnD5kyBjIHeF/view?usp=sharing).

## Features

- Utilizes eight drones for multi-angle data capture
- Integrates with Unreal Engine using the AirSim plugin
- Generates 2D bounding box matchings for pedestrians
- Produces camera calibration data (intrinsic and extrinsic parameters)
- Creates Probabilistic Occupancy Map (POM) for each timestep
- Visualizes the grid system overlaid on captured images

## Prerequisites

- Unreal Engine (version used in your project)
- AirSim plugin for Unreal Engine
- Python 3.x
- Required Python libraries: 
  - airsim
  - numpy
  - opencv-python (cv2)
  - matplotlib
  - Pillow (PIL)

## Project Structure

- `generateMatchings.py`: Generates 2D and 3D matchings for planes
- `getPedestrian.py`: Captures pedestrian data and generates matchings
- `generatePOM.py`: Creates Probabilistic Occupancy Maps
- `generateAnnotation.py`: Generates annotations from the collected data
- `grid_visualise.py`: Visualizes the grid system on captured images
- `unitConversion.py`: Utility functions for coordinate conversions

## Setup

1. Install Unreal Engine and set up your environment.
2. Install the AirSim plugin in your Unreal Engine project.
3. Clone this repository to your local machine.
4. Install required Python libraries:
   ```
   pip install airsim numpy opencv-python matplotlib Pillow
   ```
5. Configure your Unreal Engine environment to match the expected setup (six drones, pedestrians, etc.).

## Usage

1. Start your Unreal Engine environment.

2. Run the main scripts in the following order:

   a. Generate matchings for planes:
   ```
   python generateMatchings.py
   ```

   b. Capture pedestrian data:
   ```
   python getPedestrian.py
   ```

   c. Generate Probabilistic Occupancy Maps:
   ```
   python generatePOM.py
   ```

   d. Generate annotations:
   ```
   python generateAnnotation.py
   ```

3. (Optional) Visualize the grid system:
   ```
   python grid_visualise.py
   ```

## Output

The system generates several types of output:

- Images from each drone's camera
- 2D and 3D matchings for planes and pedestrians
- Camera calibration data (intrinsic and extrinsic parameters)
- Probabilistic Occupancy Maps (POMs)
- Annotations for detected objects

All output files are saved in their respective directories within the project folder.

## Customization

You can customize various parameters in the scripts:

- `datasetParameters.py`: Adjust map dimensions, camera settings, etc.
- `generateMatchings.py`: Modify drone names, camera settings, or add additional data collection steps
- `getPedestrian.py`: Adjust pedestrian detection parameters or add new features
- `generatePOM.py`: Fine-tune POM generation settings

## Troubleshooting

- Ensure that the Unreal Engine environment is running before executing the Python scripts.
- Check that the AirSim plugin is properly installed and configured in your Unreal Engine project.
- Verify that the IP address and port in `getPedestrian.py` match your Unreal Engine server settings.

## Contributing

Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## License

[Specify your license here]

## Acknowledgments

- Microsoft for the AirSim plugin
- [Any other acknowledgments or credits]
