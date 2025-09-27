#!/usr/bin/env python3
import os
import argparse
import glob


def process_label_file(file_path, target_class):
    """
    Xử lý một file label bằng cách thay thế số đầu tiên của mỗi dòng thành target_class
    
    Args:
        file_path (str): Đường dẫn đến file label
        target_class (int): Giá trị sẽ thay thế số đầu tiên
    """
    # Đọc nội dung file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Xử lý từng dòng
    processed_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:  # Đảm bảo định dạng file đúng
            # Thay thế số đầu tiên bằng target_class
            parts[0] = str(target_class)
            processed_lines.append(' '.join(parts) + '\n')
        else:
            # Giữ nguyên dòng nếu không đúng định dạng
            processed_lines.append(line)
    
    # Ghi nội dung đã xử lý trở lại file
    with open(file_path, 'w') as f:
        f.writelines(processed_lines)


def process_directory(work_dir, target_class):
    """
    Xử lý tất cả các file trong thư mục labels của các thư mục train, valid và test
    
    Args:
        work_dir (str): Thư mục làm việc chứa các thư mục train, valid, test
        target_class (int): Giá trị sẽ thay thế số đầu tiên
    """
    # Danh sách các thư mục cần xử lý
    subfolders = ['train', 'valid', 'test']
    
    total_files = 0
    processed_files = 0
    
    for subfolder in subfolders:
        labels_dir = os.path.join(work_dir, subfolder, 'labels')
        
        # Kiểm tra xem thư mục labels có tồn tại không
        if not os.path.isdir(labels_dir):
            print(f"Thư mục {labels_dir} không tồn tại, bỏ qua.")
            continue
        
        # Lấy tất cả các file trong thư mục labels
        label_files = glob.glob(os.path.join(labels_dir, '*'))
        
        total_files += len(label_files)
        
        # Xử lý từng file
        for file_path in label_files:
            if os.path.isfile(file_path):
                process_label_file(file_path, target_class)
                processed_files += 1
                print(f"Đã xử lý file: {file_path}")
    
    print(f"\nĐã hoàn thành xử lý {processed_files}/{total_files} file label.")


def main():
    # Thiết lập parser tham số dòng lệnh
    parser = argparse.ArgumentParser(description='Xử lý file labels bằng cách thay đổi số đầu tiên của mỗi dòng.')
    parser.add_argument('--work_dir', required=True, help='Thư mục làm việc chứa các thư mục train, valid, test')
    parser.add_argument('--target_class', type=int, default=0, help='Giá trị sẽ thay thế số đầu tiên (mặc định: 0)')
    
    # Parse các tham số
    args = parser.parse_args()
    
    # Kiểm tra work_dir có tồn tại không
    if not os.path.isdir(args.work_dir):
        print(f"Lỗi: Thư mục {args.work_dir} không tồn tại.")
        return
    
    print(f"Bắt đầu xử lý với work_dir={args.work_dir}, target_class={args.target_class}")
    
    # Xử lý tất cả các file trong các thư mục
    process_directory(args.work_dir, args.target_class)


if __name__ == "__main__":
    main()