# MATRIX (Multi-Aerial TRacking In compleX environments) Dataset Generator

## Overview

This project implements a multi-camera multiview system designed to generate detection and tracking datasets using eight drones in an Unreal Engine environment. The drones, controlled via the AirSim plugin by Microsoft, fly randomly and capture images, creating a rich dataset for computer vision tasks.

![](https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExNjRqdDM3eHZtdnh0MTMxM2Eyc2k2MnQwODB5ZG4wZnBsYW1odGFqMyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/JKrospdoZH6rpza6tS/giphy.gif)

## Download

The dataset with 1000 timesteps, 40 pedestrians on a 30m x 30m area can be downloaded [here](https://drive.google.com/file/d/1hSB72MSPQLEIL-9Hb0DoBnD5kyBjIHeF/view?usp=sharing).

## Features

- Utilizes eight drones for multi-angle data capture
- Integrates with Unreal Engine using the AirSim plugin
- Generates 2D bounding box matchings for pedestrians
- Real-time Line of Sight (LoS) calculation between drones and subjects
- Produces camera calibration data (intrinsic and extrinsic parameters)
- Creates Probabilistic Occupancy Map (POM) for each timestep
- Visualizes the grid system overlaid on captured images. Code adapted [from](https://github.com/hou-yz/MVDet/blob/master/grid_visualize.py)

## Project Structure

- `generateMatchings.py`: Calibrates each drone camera using a checkerboard pattern in the UE enviroment
- `getPedestrian.py`: Captures pedestrian data
- `pedestrianLoS.py`: Checks if pedestrian is within LoS of drone 
- `generatePOM.py`: Creates Probabilistic Occupancy Maps
- `generateAnnotation.py`: Generates annotations from the collected data
- `grid_visualise.py`: Visualizes the grid system on captured images
- `unitConversion.py`: Utility functions for coordinate conversions

## Description of Available Files

The dataset provides:

1. **Synchronized Frames**
   - Extracted at 2 FPS
   - Full HD resolution (1920×1080)

2. **Calibration Files**
   - Uses the Pinhole camera model
   - Compatible with OpenCV library
   - Both intrinsic and extrinsic calibrations included
   - Updated for each timestep to account for drone movement

3. **Ground Truth Annotations**
   - Provided in JSON format
   - Contains pedestrian positions and bounding boxes
   - Line-of-Sight (LoS) information for each camera
   - Probabilistic Occupancy Maps (POMs)

4. **Position Data**
   - World coordinate system mappings
   - Grid-based position tracking
   - Conversion utilities between coordinate systems
     
## Dataset Statistics

For the standard configuration (1000 timesteps):
- Number of pedestrians: 40
- Coverage area: 30m x 30m
- Total number of frames: 8,000 (1,000 per camera)

## Dataset Generation

To generate your own version of the dataset:

1. **Environment Setup**
   - Unreal Engine 5.2
   - AirSim/Colosseum plugin ([download](https://github.com/CodexLabsLLC/Colosseum))
   - Python 3.10 with required packages

2. **Required Python Packages**
   ```bash
   pip install airsim numpy opencv-python matplotlib Pillow
   ```

3. **Generation Process**
   ```bash
   # Just run main.py
   python main.py
   ```


4. (Optional) Visualize the grid system:
   ```
   python grid_visualise.py
   ```

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

## Contact

For access to the Unreal Engine environment or questions about the dataset:
- Email: kosta.dakic@outlook.com

## License

[Specify your license here]

## Citation

If you use this dataset in your research, please cite:
[Add citation information]

## Acknowledgments

- Microsoft for the [AirSim](https://microsoft.github.io/AirSim/api_docs/html/#) plugin
- [MultiviewX toolkit](https://github.com/hou-yz/MultiviewX)
- [Colosseum](https://github.com/CodexLabsLLC/Colosseum) Open source simulator for autonomous robotics
- [MVDet](https://github.com/hou-yz/MVDet/tree/master)
- [WILDTRACK](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/)
