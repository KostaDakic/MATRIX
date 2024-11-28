import os
import re
import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from unitConversion import *

def read_pom(fpath):
    # This function remains unchanged
    bbox_by_pos_cam = {}
    cam_pos_pattern = re.compile(r'(\d+) (\d+)')
    cam_pos_bbox_pattern = re.compile(r'(\d+) (\d+) ([-\d]+) ([-\d]+) (\d+) (\d+)')
    with open(fpath, 'r') as fp:
        for line in fp:
            if 'RECTANGLE' in line:
                cam, pos = map(int, cam_pos_pattern.search(line).groups())
                if pos not in bbox_by_pos_cam:
                    bbox_by_pos_cam[pos] = {}
                if 'notvisible' in line:
                    bbox_by_pos_cam[pos][cam] = [-1, -1, -1, -1]
                else:
                    cam, pos, left, top, right, bottom = map(int, cam_pos_bbox_pattern.search(line).groups())
                    bbox_by_pos_cam[pos][cam] = [left, top, right, bottom]
    return bbox_by_pos_cam

def read_gt(file_path):
    frames = []
    current_frame = []
    frame_number = 0

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split()
                pid, pos_id, x, y, z = map(float, parts)
                current_frame.append([frame_number, pid, pos_id, x, y, z])
            else:
                if current_frame:
                    frames.append(np.array(current_frame))
                    current_frame = []
                    frame_number += 1

    if current_frame:
        frames.append(np.array(current_frame))

    return frames

def create_pid_annotation(pid, pos, bbox_by_pos_cam):
    person_annotation = {'personID': int(pid), 'positionID': int(pos), 'views': []}
    for cam in range(len(bbox_by_pos_cam[pos])):
        bbox = bbox_by_pos_cam[pos][cam]
        # Reverse the x-axis (1920 - x)
        xmin_reversed = 1920 - bbox[2]  # right becomes left
        xmax_reversed = 1920 - bbox[0]  # left becomes right
        ymin_reversed = bbox[1]
        ymax_reversed = bbox[3]

        view_annotation = {
            'viewNum': cam,
            'xmin': int(xmin_reversed),
            'ymin': int(ymin_reversed),
            'xmax': int(xmax_reversed),
            'ymax': int(ymax_reversed)
        }
        person_annotation['views'].append(view_annotation)
    return person_annotation

def annotate(timestep, max_timestep):
    os.makedirs('annotations_positions', exist_ok=True)

    # Read data from all cameras
    file_path = f'matchings/Pedestrians/3d_{timestep:04d}.txt'
    gt_frames = read_gt(file_path)
    all_gt_frames = gt_frames

    # Assume all cameras have the same number of frames
    num_ped = len(all_gt_frames[0])

    # Read the corresponding POM file for this frame
    pom_file = f'POMs/rectangles_{timestep:04d}.pom'
    bbox_by_pos_cam = read_pom(pom_file)

    annotations = []
    for i in range(num_ped):
        _, pid, pos_id, x, y, _ = all_gt_frames[0][i]
        # Only include pedestrians whose pos_id is in bbox_by_pos_cam
        if int(pos_id) in bbox_by_pos_cam:
            annotations.append(create_pid_annotation(pid, pos_id, bbox_by_pos_cam))

    with open(f'annotations_positions/{timestep:04d}.json', 'w') as fp:
        json.dump(annotations, fp, indent=4)

    # Visualize
    if timestep == 0 or timestep == max_timestep-1:
        for cam in range(NUM_CAM):
            img = Image.open(f'Image_subsets/D{cam + 1}/{timestep:04d}.png')
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            for anno in annotations:
                anno = anno['views'][cam]
                bbox = tuple([anno['xmin'], anno['ymin'], anno['xmax'], anno['ymax']])
                if bbox[0] == -1 and bbox[1] == -1:
                    continue
                cv2.rectangle(img, bbox[:2], bbox[2:], (0, 255, 0), 2)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img.save(f'bbox_cam{cam + 1}.png')
            plt.imshow(img)
            plt.show()

if __name__ == '__main__':
    max_timestep = 1
    for timestep in range(max_timestep):
        annotate(timestep, max_timestep)