import os
import argparse
import globl
def process_label_file(file_path):
    """
    Process a single label file:
    - Remove lines where the first number is 0
    - Convert lines where the first number is 1 to start with 0
    """
    # Read the original file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Process lines
    processed_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 0:
            # If id is 1, change to 0
            if parts[0] == '1':
                parts[0] = '0'
                processed_lines.append(' '.join(parts) + '\n')
            # If id is 0, skip the line (effectively deleting it)
    
    # Write processed lines back to the same file
    with open(file_path, 'w') as f:
        f.writelines(processed_lines)

def process_labels_in_directory(work_dir):
    """
    Process label files in train, valid, and test directories
    """
    # Directories to process
    subdirs = ['train', 'valid', 'test']
    
    for subdir in subdirs:
        # Full path to labels directory
        labels_dir = os.path.join(work_dir, subdir, 'labels')
        
        # Check if labels directory exists
        if not os.path.exists(labels_dir):
            print(f"Warning: Directory {labels_dir} does not exist. Skipping.")
            continue
        
        # Find all label files (assuming .txt extension)
        label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
        
        # Process each label file
        for file_path in label_files:
            process_label_file(file_path)
            print(f"Processed: {file_path}")

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Process label files in dataset directories.')
    parser.add_argument('work_dir', type=str, help='Root directory containing train, valid, test folders')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate work directory
    if not os.path.isdir(args.work_dir):
        print(f"Error: {args.work_dir} is not a valid directory.")
        return
    
    # Process labels
    process_labels_in_directory(args.work_dir)
    print("Label processing completed.")

if __name__ == '__main__':
    main()