import re
import csv
import sys

def extract_data_from_logs(log_content):
    """
    Extract epoch, resolution, and mAP data from log content.
    """
    data = []
    
    # Regular expression to match the log format
    pattern = r'tmp_results/epoch(\d+)_res(\d+)\.log:mAP: ([\d\.]+)'
    
    # Process each line
    for line in log_content.strip().split('\n'):
        match = re.match(pattern, line)
        if match:
            epoch = int(match.group(1))
            resolution_index = int(match.group(2))
            map_value = float(match.group(3))
            
            data.append((resolution_index, epoch, map_value))
    
    # Sort by resolution_index, then epoch for consistent ordering
    data.sort()
    
    return data

def create_csv_file(data, output_filename="results.csv"):
    """
    Create a CSV file with the format: resolution_index, num_epochs, mAP
    """
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(["resolution_index", "num_epochs", "mAP"])
        
        # Write data
        for resolution_index, epoch, map_value in data:
            writer.writerow([resolution_index, epoch, map_value])
    
    print(f"CSV file created successfully: {output_filename}")

# Example usage
if __name__ == "__main__":
    # Uncomment the next 3 lines if reading from a file instead
    with open(sys.argv[1], 'r') as file:
         log_content = file.read()
    
    # Extract data from logs
    data = extract_data_from_logs(log_content)
    
    # Create CSV file
    create_csv_file(data, "resolution_map_results.csv")
