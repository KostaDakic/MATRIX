import os
import glob

def process_los_files(directory):
    # Get all txt files in the LoS directory
    file_pattern = os.path.join(directory, 'Pedestrians', 'LoS', '*.txt')
    files = glob.glob(file_pattern)

    for file_path in files:
        process_file(file_path)

def process_file(file_path):
    temp_file_path = file_path + '.temp'

    with open(file_path, 'r') as input_file, open(temp_file_path, 'w') as output_file:
        for line in input_file:
            parts = line.split()
            if len(parts) >= 5:
                x, y = float(parts[2]), float(parts[3])
                if x <= 30 and y <= 30:
                    output_file.write(line)

    # Replace the original file with the processed file
    os.replace(temp_file_path, file_path)
    print(f"Processed: {file_path}")

if __name__ == '__main__':
    matchings_directory = 'matchings'  # Adjust this path if needed
    process_los_files(matchings_directory)