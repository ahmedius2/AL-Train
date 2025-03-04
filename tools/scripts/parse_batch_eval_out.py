import os
import re
import csv
import glob
import sys

def extract_metrics_from_log_file(file_path):
    """
    Extract epoch, resolution, mAP and NDS values from a log file.
    """
    # Extract epoch and resolution from filename
    filename = os.path.basename(file_path)
    match = re.match(r'epoch(\d+)_res(\d+)\.log', filename)
    
    if not match:
        return None
    
    epoch = int(match.group(1))
    resolution_index = int(match.group(2))
    
    # Read the file content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Extract mAP and NDS using regex
    map_match = re.search(r'mAP: ([\d\.]+)', content)
    nds_match = re.search(r'NDS: ([\d\.]+)', content)
    
    if not map_match or not nds_match:
        return None
    
    map_value = float(map_match.group(1))
    nds_value = float(nds_match.group(1))
    
    return {
        'resolution_index': resolution_index,
        'num_epochs': epoch,
        'mAP': map_value,
        'NDS': nds_value
    }

def process_log_directory(directory_path):
    """
    Process all log files in a directory and create a CSV file with metrics.
    """
    # Find all log files matching the pattern
    log_files = glob.glob(os.path.join(directory_path, "epoch*_res*.log"))
    
    if not log_files:
        print(f"No matching log files found in {directory_path}")
        return
    
    data = []
    
    # Process each log file
    for log_file in log_files:
        result = extract_metrics_from_log_file(log_file)
        if result:
            data.append(result)
    
    # Sort by resolution_index, then epoch for consistent ordering
    data.sort(key=lambda x: (x['resolution_index'], x['num_epochs']))
    
    output_filename = directory_path.split('_')[0] + "results.csv"
    # Write to CSV
    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = ['resolution_index', 'num_epochs', 'mAP', 'NDS']
        print(','.join(fieldnames))
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            row_str = [str(v) for v in row.values()]
            print(','.join(row_str))
            writer.writerow(row)

    print(f"CSV file created successfully: {output_filename}")
    print(f"Processed {len(data)} log files")

if __name__ == "__main__":
    # Specify the directory containing the log files
    log_directory = sys.argv[1]  # Change this to your directory path
    
    # Process the directory
    process_log_directory(log_directory)
