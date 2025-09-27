import os
import cv2

# Đường dẫn tới thư mục ảnh, nhãn và thư mục debug
images_folder = 'images'
labels_folder = 'labels'
debug_folder = 'debug_folder'

# Tạo thư mục debug nếu chưa tồn tại
os.makedirs(debug_folder, exist_ok=True)

# Lặp qua tất cả các ảnh trong thư mục images
for image_file in os.listdir(images_folder):
    # Chỉ xử lý các file có đuôi ảnh
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(images_folder, image_file)
        label_path = os.path.join(labels_folder, os.path.splitext(image_file)[0] + '.txt')

        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            print(f"Lỗi đọc ảnh: {image_path}")
            continue

        height, width = image.shape[:2]

        # Nếu có file label tương ứng
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    # Bỏ qua class_id, lấy x_center, y_center, w, h đã chuẩn hóa
                    x_center, y_center, w, h = map(float, parts[1:])

                    # Chuyển từ chuẩn hóa sang pixel
                    x_center *= width
                    y_center *= height
                    w *= width
                    h *= height

                    # Tính toán toạ độ góc trái trên và góc phải dưới
                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    x2 = int(x_center + w / 2)
                    y2 = int(y_center + h / 2)

                    # Vẽ hình chữ nhật đỏ
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Lưu ảnh đã vẽ bounding box vào debug_folder
        debug_image_path = os.path.join(debug_folder, image_file)
        cv2.imwrite(debug_image_path, image)

print("Đã hoàn tất quá trình tạo ảnh debug với bounding boxes.")
