#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image Augmentation Tool for Object Detection

This script augments images by adding additional objects from a separate folder
with various transformations (rotation, scaling) while properly handling bounding boxes.
"""

import os
import cv2
import numpy as np
import random
import argparse
import logging
import shutil
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import math
from tqdm import tqdm
import yaml
count_none = 0
count_edge = 0
count_ovl = 0

def load_config(config_path='cfg_augment.yaml'):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

# Load the configuration
config = load_config()
image_config = config['add_images']

# Access image addition parameters
MIN_ADD_IMAGES = image_config['min_count']
MAX_ADD_IMAGES = image_config['max_count']
MIN_ROTATE = image_config['min_rotate']
MAX_ROTATE = image_config['max_rotate']
MIN_ZOOM = image_config['min_zoom']
MAX_ZOOM = image_config['max_zoom']
MIN_VISIBILITY = image_config['min_visibility']
MAX_VISIBILITY = image_config['max_visibility']
MAX_ATTEMPTS = image_config['max_attempts']
data_path = image_config['data_path']
output_dir = image_config['output_dir']
NONE_RATIO = image_config['none_ratio']
EDGE_RATIO = image_config['edge_ratio']
OVL_RATIO = image_config['ovl_ratio']


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("augmentation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ImageAugmentation")

@dataclass
class BoundingBox:
    """Class for representing a bounding box in normalized coordinates."""
    class_id: int
    x_center: float  # Normalized x center coordinate
    y_center: float  # Normalized y center coordinate
    width: float     # Normalized width
    height: float    # Normalized height
    
    def get_absolute_coords(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert normalized coordinates to absolute pixel coordinates."""
        x1 = int((self.x_center - self.width / 2) * img_width)
        y1 = int((self.y_center - self.height / 2) * img_height)
        x2 = int((self.x_center + self.width / 2) * img_width)
        y2 = int((self.y_center + self.height / 2) * img_height)
        return x1, y1, x2, y2
    
    def get_area(self) -> float:
        """Get the area of the bounding box in normalized units."""
        return self.width * self.height
    
    @classmethod
    def from_absolute_coords(cls, class_id: int, x1: int, y1: int, x2: int, y2: int, 
                           img_width: int, img_height: int) -> 'BoundingBox':
        """Create a BoundingBox from absolute pixel coordinates."""
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        x_center = (x1 / img_width) + (width / 2)
        y_center = (y1 / img_height) + (height / 2)
        return cls(class_id, x_center, y_center, width, height)
    
    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if this bounding box intersects with another one."""
        # Calculate the coordinates of the boxes
        this_x1 = self.x_center - self.width / 2
        this_y1 = self.y_center - self.height / 2
        this_x2 = self.x_center + self.width / 2
        this_y2 = self.y_center + self.height / 2
        
        other_x1 = other.x_center - other.width / 2
        other_y1 = other.y_center - other.height / 2
        other_x2 = other.x_center + other.width / 2
        other_y2 = other.y_center + other.height / 2
        
        # Check if one box is to the left of the other
        if this_x2 < other_x1 or other_x2 < this_x1:
            return False
        
        # Check if one box is above the other
        if this_y2 < other_y1 or other_y2 < this_y1:
            return False
        
        return True
    
    def intersection_area(self, other: 'BoundingBox') -> float:
        """Calculate the intersection area with another box."""
        # Calculate the coordinates of the boxes
        this_x1 = self.x_center - self.width / 2
        this_y1 = self.y_center - self.height / 2
        this_x2 = self.x_center + self.width / 2
        this_y2 = self.y_center + self.height / 2
        
        other_x1 = other.x_center - other.width / 2
        other_y1 = other.y_center - other.height / 2
        other_x2 = other.x_center + other.width / 2
        other_y2 = other.y_center + other.height / 2
        
        # Calculate intersection coordinates
        inter_x1 = max(this_x1, other_x1)
        inter_y1 = max(this_y1, other_y1)
        inter_x2 = min(this_x2, other_x2)
        inter_y2 = min(this_y2, other_y2)
        
        # Return 0 if there is no intersection
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0
        
        # Calculate and return intersection area
        return (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    def is_inside_image(self, margin: float = 0.0) -> bool:
        """Check if the bounding box is completely inside the image with optional margin."""
        return (margin <= self.x_center - self.width / 2 and 
                self.x_center + self.width / 2 <= 1 - margin and
                margin <= self.y_center - self.height / 2 and 
                self.y_center + self.height / 2 <= 1 - margin)
    
    def visibility_ratio(self) -> float:
        """Calculate how much of the box is inside the image (0.0 to 1.0)."""
        x1 = max(0, self.x_center - self.width / 2)
        y1 = max(0, self.y_center - self.height / 2)
        x2 = min(1, self.x_center + self.width / 2)
        y2 = min(1, self.y_center + self.height / 2)
        
        if x1 >= x2 or y1 >= y2:
            return 0.0
        
        visible_area = (x2 - x1) * (y2 - y1)
        original_area = self.width * self.height
        
        if original_area == 0:
            return 0.0
        
        return visible_area / original_area
    
    def to_yolo_format(self) -> str:
        """Convert the bounding box to YOLO format string."""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


class ImageAugmenter:
    """Class for augmenting images with additional objects."""
    
    def __init__(self, 
                 images_folder: str, 
                 labels_folder: str, 
                 add_images_folder: str,
                 output_folder: str,
                 min_rotate: float = MIN_ROTATE,
                 max_rotate: float = MAX_ROTATE,
                 min_zoom: float = MIN_ZOOM,
                 max_zoom: float = MAX_ZOOM,
                 min_visibility: float = MIN_VISIBILITY,
                 max_visibility: float = MAX_VISIBILITY,
                 max_attempts: int = MAX_ATTEMPTS):
        """
        Initialize the ImageAugmenter.
        
        Args:
            images_folder: Path to the folder containing original images
            labels_folder: Path to the folder containing original labels
            add_images_folder: Path to the folder containing images to add
            output_folder: Path to the folder where augmented images and labels will be saved
            min_rotate: Minimum rotation angle in degrees
            max_rotate: Maximum rotation angle in degrees
            min_zoom: Minimum zoom factor
            max_zoom: Maximum zoom factor
            full_ratio: Probability that the added image must be fully inside
            min_visibility: Minimum required visibility ratio
            max_attempts: Maximum number of attempts to place an object
        """
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        self.add_images_folder = add_images_folder
        self.output_folder = output_folder
        self.min_rotate = min_rotate
        self.max_rotate = max_rotate
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.min_visibility = min_visibility
        self.max_visibility = max_visibility
        self.max_attempts = max_attempts
        
        # Create output folders if they don't exist
        os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "labels"), exist_ok=True)
        
        # Load add images
        self.add_images = self._load_add_images()
        if not self.add_images:
            logger.error(f"No images found in {add_images_folder}")
            raise ValueError(f"No images found in {add_images_folder}")
        
        logger.info(f"Loaded {len(self.add_images)} images from {add_images_folder}")
    
    def _load_add_images(self) -> List[str]:
        """Load all images from the add_images folder."""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        return [f for f in os.listdir(self.add_images_folder) 
                if os.path.isfile(os.path.join(self.add_images_folder, f)) and 
                os.path.splitext(f)[1].lower() in valid_extensions]
    
    def _load_label(self, label_path: str) -> List[BoundingBox]:
        """Load bounding boxes from a label file."""
        bboxes = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        bboxes.append(BoundingBox(class_id, x_center, y_center, width, height))
            return bboxes
        except Exception as e:
            logger.error(f"Error loading label file {label_path}: {e}")
            return []
    
    def _save_label(self, label_path: str, bboxes: List[BoundingBox]) -> None:
        """Save bounding boxes to a label file."""
        try:
            with open(label_path, 'w') as f:
                for bbox in bboxes:
                    f.write(f"{bbox.to_yolo_format()}\n")
        except Exception as e:
            logger.error(f"Error saving label file {label_path}: {e}")
    
    def _rotate_point(self, x: float, y: float, cx: float, cy: float, angle_rad: float) -> Tuple[float, float]:
        """Rotate a point around a center point by the given angle."""
        x_shifted = x - cx
        y_shifted = y - cy
        
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        
        x_rotated = cos_angle * x_shifted - sin_angle * y_shifted
        y_rotated = sin_angle * x_shifted + cos_angle * y_shifted
        
        return x_rotated + cx, y_rotated + cy

    def _get_rotated_box(self, bbox: BoundingBox, angle_deg: float) -> BoundingBox:
        """
        Get a bounding box that encloses the rotated original bounding box.
        
        Args:
            bbox: The original bounding box
            angle_deg: Rotation angle in degrees
            
        Returns:
            A new bounding box that contains the rotated box
        """
        # Convert angle to radians
        angle_rad = math.radians(angle_deg)
        
        # Get the four corners of the box
        half_w = bbox.width / 2
        half_h = bbox.height / 2
        cx, cy = bbox.x_center, bbox.y_center
        
        # Calculate the four corners
        corners = [
            (cx - half_w, cy - half_h),  # top-left
            (cx + half_w, cy - half_h),  # top-right
            (cx + half_w, cy + half_h),  # bottom-right
            (cx - half_w, cy + half_h)   # bottom-left
        ]
        
        # Rotate each corner
        rotated_corners = [self._rotate_point(x, y, cx, cy, angle_rad) for x, y in corners]
        
        # Find the min/max x and y coordinates to form the new box
        min_x = min(x for x, _ in rotated_corners)
        max_x = max(x for x, _ in rotated_corners)
        min_y = min(y for _, y in rotated_corners)
        max_y = max(y for _, y in rotated_corners)
        
        # Calculate the new width, height, and center
        new_width = max_x - min_x
        new_height = max_y - min_y
        new_cx = min_x + new_width / 2
        new_cy = min_y + new_height / 2
        
        return BoundingBox(bbox.class_id, new_cx, new_cy, new_width, new_height)
    
    def _place_object(self, 
                     base_img: np.ndarray, 
                     add_img: np.ndarray,
                     base_bboxes: List[BoundingBox], 
                     angle: float, 
                     target_height: float,
                     target_width: float,
                     processing_type: str) -> Tuple[np.ndarray, Optional[BoundingBox]]:
        """
        Place an object on the base image with proper transformations.
        
        Args:
            base_img: The base image
            add_img: The image to add
            base_bboxes: Existing bounding boxes in the base image
            is_full: Whether the added object must be fully inside
            angle: Rotation angle in degrees
            zoom: Zoom factor
            
        Returns:
            Tuple of (augmented image, new bounding box or None if placement failed)
        """
        base_height, base_width = base_img.shape[:2]
        add_height, add_width = add_img.shape[:2]

        # print("target_area", target_area)
        # apply rotate
        
        # Apply zoom
        # new_width = int(add_width * zoom)
        # new_height = int(add_height * zoom)
        # if new_width <= 0 or new_height <= 0:
        #     logger.warning(f"Invalid dimensions after zoom: {new_width}x{new_height}, skipping")
        #     return base_img.copy(), None
        
        # resized_img = cv2.resize(add_img, (new_width, new_height))
        # apply zoom\
        target_area = target_width * target_height

        # zoom = target_area / (add_width * add_height)
        zoom = max(target_width, target_height) / min(add_width, add_height)
        zoom *= random.uniform(MIN_ZOOM, MAX_ZOOM)

        zoomed_height = int(add_height * zoom)
        zoomed_width = int(add_width * zoom)
        zoomed_img = cv2.resize(add_img, (zoomed_width, zoomed_height), interpolation=cv2.INTER_LINEAR)

        # print("target_area", target_area)
        # print("add_width", add_width)
        # print("add_height", add_height)
        # print("zoomed_width", zoomed_width)
        # print("zoomed_height", zoomed_height)
        # Apply rotation
        center = (zoomed_width // 2, zoomed_height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions after rotation
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        rotated_width = int(zoomed_width * cos_angle + zoomed_height * sin_angle)
        rotated_height = int(zoomed_width * sin_angle + zoomed_height * cos_angle)
        
        # Adjust rotation matrix
        rotation_matrix[0, 2] += rotated_width / 2 - center[0]
        rotation_matrix[1, 2] += rotated_height / 2 - center[1]
        
        # Apply rotation
        rotated_img = cv2.warpAffine(zoomed_img, rotation_matrix, (rotated_width, rotated_height))
        

        # # Create a mask for the rotated image (for transparent pasting)
        # if rotated_img.shape[2] == 4:  # RGBA image
        #     alpha_mask = rotated_img[:, :, 3] / 255.0
        #     rgb_img = rotated_img[:, :, :3]
        # else:
        #     # If no alpha channel, create a mask from non-black pixels
        #     gray_img = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
        #     _, alpha_mask = cv2.threshold(gray_img, 1, 1, cv2.THRESH_BINARY)
        #     rgb_img = rotated_img
            
        # Create the bounding box for the rotated image
        new_width = rotated_width / base_width
        new_height = rotated_height / base_height
        original_bbox = BoundingBox(0, 0.5, 0.5, 1.0, 1.0)  # Full add image
        new_bbox = BoundingBox(0, 0.5, 0.5, new_width, new_height)
        new_img = rotated_img
        # Scale the bounding box according to the zoom factor

        # TODO: need to recheck
        # rotated_bbox.width *= zoom
        # rotated_bbox.height *= zoom
        
        # For up to max_attempts, try to place the image
        for _ in range(self.max_attempts):
            # Determine placement position
            new_x_center = random.uniform(0, 1)
            new_y_center = random.uniform(0, 1)
            temp_bbox = BoundingBox(0, new_x_center, new_y_center, new_width , new_height)

            if processing_type == "none":
                # if temp_bbox not fully inside the image, continue
                if not temp_bbox.is_inside_image():
                    continue
                # if temp_bbox touching other bboxes, continue
                touch = False
                for bbox in base_bboxes:
                    if bbox.intersects(temp_bbox):
                        touch = True
                        break
                if touch:
                    continue

            elif processing_type == "edge":
                # if temp_bbox fully inside the image, continue
                if temp_bbox.is_inside_image():
                    continue
                # if temp_bbox touching upper edge, continue
                if temp_bbox.y_center - temp_bbox.height / 2 < 0:
                    continue
                # if temp_bbox touching left edge, continue
                if temp_bbox.x_center - temp_bbox.width / 2 < 0:
                    continue
                # if temp_bbox touching right edge, continue
                if temp_bbox.x_center + temp_bbox.width / 2 > 1:
                    continue
                # if temp_bbox not touching other bboxes, continue
                touch = False
                for bbox in base_bboxes:
                    if bbox.intersects(temp_bbox):
                        touch = True
                        break
                if touch:
                    continue
                # if visible area is less than min_visibility or more than max_visibility, continue
                # if temp_bbox.visibility_ratio() < self.min_visibility:
                #     continue
            else:
                # if temp_bbox not fully inside the image, continue
                if not temp_bbox.is_inside_image():
                    continue
                # if temp_bbox not touching other bboxes, continue
                touch = False
                count_touch = 0
                for bbox in base_bboxes:
                    if bbox.intersects(temp_bbox):
                        touch = True
                        count_touch += 1
                        break
                if count_touch > 1:
                    continue
                if not touch:
                    continue
                # if temp_bbox touch any center of other bboxes, continue
                touch_center = False
                for bbox in base_bboxes:
                    # check if new_bbox contain center of bbox
                    if(bbox.x_center > temp_bbox.x_center - temp_bbox.width /2 and
                        bbox.x_center < temp_bbox.x_center + temp_bbox.width / 2 and
                        bbox.y_center > temp_bbox.y_center - temp_bbox.height /2 and
                        bbox.y_center < temp_bbox.y_center + temp_bbox.height / 2):
                        touch_center = True
                        break
                    # check if bbox contain center of new_bbox
                    if(temp_bbox.x_center > bbox.x_center - bbox.width /2 and
                        temp_bbox.x_center < bbox.x_center + bbox.width / 2 and
                        temp_bbox.y_center > bbox.y_center - bbox.height /2 and
                        temp_bbox.y_center < bbox.y_center + bbox.height / 2):
                        touch_center = True
                        break
                if touch_center:
                    continue

            new_bbox = temp_bbox
            break
        
        else:
            # If we've exhausted all attempts
            logger.warning("Failed to place object after maximum attempts")
            return base_img.copy(), None
        
        # Create a copy of the base image to avoid modifying the original
        augmented_img = base_img.copy()
        

        orig = augmented_img
        child = new_img
        padding = 500;
        # Add padding to the original image
        orig = cv2.copyMakeBorder(orig, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0,0,0))
        # debug to folder debug1 for augmented_img and rotated_img
        # step = 0
        # rnd_id = random.randint(0, 10000)
        # cv2.imwrite(f"debug1/{step}_{rnd_id}_base.png", orig)
        # cv2.imwrite(f"debug1/{step}_{rnd_id}_add.png", child)
        # step += 1
        child_rgb = child[..., :3]
        alpha = child[..., 3] / 255.0
        
        # Feather alpha mask
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        
        # Remove RGB from nearly transparent regions
        child_rgb[alpha < 0.05] = 0
        alpha[alpha < 0.05] = 0

        # Get size and center position
        h, w = child_rgb.shape[:2]
        center = (int(new_bbox.x_center * base_width), int(new_bbox.y_center * base_height))
        top_left = (max(0, center[0] - w // 2), max(0, center[1] - h // 2))

        top_left = top_left[0] + padding, top_left[1] + padding

        # center = (orig.shape[1] // 2, orig.shape[0] // 2)
        # top_left = (center[0] - w // 2, center[1] - h // 2)
        

        # Crop background
        crop = orig[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
        if crop.shape[:2] != (h, w):
            crop = cv2.resize(crop, (w, h))
        
        # Texture: Laplacian variance
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
        adjust_texture = np.clip(500 / (lap_var + 1e-5), 0.8, 1.1)
        
        # Brightness map
        brightness = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bright_mask = (brightness > 230).astype(np.float32)
        adjust_brightness = 1.05 + 0.1 * bright_mask  # Slight boost in bright areas
        
        # Combine adjustments only for visible alpha regions
        mask_visible = (alpha > 0.2).astype(np.float32)
        combined_adjust = 1 + (adjust_texture - 1) * mask_visible
        combined_adjust *= adjust_brightness
        
        # Final alpha
        alpha_dynamic = np.clip(alpha * combined_adjust, 0, 1)
        
        # Blend
        blended = (child_rgb * alpha_dynamic[..., None] + crop * (1 - alpha_dynamic[..., None])).astype(np.uint8)
        
        # Overlay back to the original image
        # output = orig.copy()

        # padding = 500
        # h, w = blended.shape[:2]

        # # Thêm padding vào output
        # output_padded = cv2.copyMakeBorder(output, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0,0,0))

        # # Điều chỉnh tọa độ top_left do đã thêm padding
        # x, y = top_left
        # x += padding
        # y += padding

        # # Tính toán kích thước vùng dán, cắt nếu vượt quá biên
        # max_y, max_x = output_padded.shape[:2]
        # blended_cropped = blended[:min(h, max_y - y), :min(w, max_x - x)]

        # # Gán vùng ảnh đã cắt vào ảnh có padding
        # output_padded[y:y+blended_cropped.shape[0], x:x+blended_cropped.shape[1]] = blended_cropped

        # # Cắt lại để bỏ phần padding, trả về kích thước như ảnh gốc
        # output_final = output_padded[padding:-padding, padding:-padding]
        output = orig.copy()
        output[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w] = blended
        output = output[padding:-padding, padding:-padding]
        # output = cv2.seamlessClone(child_rgb, orig, mask, center, cv2.NORMAL_CLONE)
        augmented_img = output

        return augmented_img, new_bbox
    
    def _find_largest_bbox_of_class(self, bboxes: List[BoundingBox], class_id: int, 
                               img_width: int, img_height: int) -> Tuple[float, float]:
        """
        Find the largest bounding box with the specified class ID in absolute pixels.
        
        Args:
            bboxes: List of bounding boxes
            class_id: The class ID to filter by
            img_width: Width of the image
            img_height: Height of the image
            
        Returns:
            Tuple of (width, height) in pixels of the largest bbox with the specified class_id
        """
        largest_area = 0
        largest_width = 0
        largest_height = 0
        
        # Filter bboxes by class_id
        class_bboxes = [bbox for bbox in bboxes if bbox.class_id == class_id]
        
        if not class_bboxes:
            # Return a default size if no bounding boxes of the specified class exist
            logger.debug(f"No bounding boxes found with class_id={class_id}, using default size")
            return img_width * 0.2, img_height * 0.2
        
        for bbox in class_bboxes:
            abs_width = bbox.width * img_width
            abs_height = bbox.height * img_height
            area = abs_width * abs_height
            
            if area > largest_area:
                largest_area = area
                largest_width = abs_width
                largest_height = abs_height
        
        logger.debug(f"Largest bbox with class_id={class_id}: {largest_width:.1f}x{largest_height:.1f} pixels")
        return largest_width, largest_height
    
    def _calculate_zoom_factor(self, add_img_width: int, add_img_height: int, 
                              target_width: float, target_height: float) -> float:
        """
        Calculate an appropriate zoom factor to make the add_image match the target size.
        
        Args:
            add_img_width: Width of the add image
            add_img_height: Height of the add image
            target_width: Target width in pixels
            target_height: Target height in pixels
            
        Returns:
            Appropriate zoom factor with some random variation
        """
        # Calculate the scale factors required to match target width and height
        width_scale = target_width / add_img_width
        height_scale = target_height / add_img_height
        
        # Take the smaller scale to ensure the object isn't too large
        base_scale = min(width_scale, height_scale)
        
        # Add some random variation (±20%)
        variation = random.uniform(MIN_ZOOM, MAX_ZOOM)
        zoom = base_scale * variation
        
        # Ensure the zoom is within the allowed range
        zoom = max(min(zoom, self.max_zoom), self.min_zoom)
        
        logger.debug(f"Calculated zoom factor: {zoom:.2f}")
        return zoom
    
    def augment_image(self, image_path: str, label_path: str, i: int, processing_type: str) -> Tuple[str, str]:
        global count_none, count_edge, count_ovl
        """
        Augment an image by adding objects from the add_images folder.
        
        Args:
            image_path: Path to the image to augment
            label_path: Path to the label file
            
        Returns:
            Tuple of (output image path, output label path)
        """
        # Load the image
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None, None
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None, None
        
        img_height, img_width = image.shape[:2]
        
        # Load the labels
        bboxes = self._load_label(label_path)
        if not bboxes:
            logger.warning(f"No bounding boxes found in {label_path}")
        
        # Find the largest bounding box with class_id=1
        # We're specifically looking for class_id 1 as per requirements
        target_width, target_height = self._find_largest_bbox_of_class(bboxes, class_id=0, 
                                                                      img_width=img_width, img_height=img_height)
        
        target_area = target_width * target_height
        # Determine how many objects to add
        # Count the number of persons, both class_id=0 and class_id=1
        num_persons = sum(1 for bbox in bboxes if bbox.class_id in [0, 1])
        # num_add = random.randint(MIN_ADD_IMAGES, max(MIN_ADD_IMAGES + 1, min(num_persons, MAX_ADD_IMAGES))) if num_persons > 0 else MIN_ADD_IMAGES
        if(MIN_ADD_IMAGES == MAX_ADD_IMAGES):
            num_add = MIN_ADD_IMAGES
        else:
            num_add = random.randint(MIN_ADD_IMAGES, MAX_ADD_IMAGES)

        # CONFIG SO TRE CON
        if processing_type == "edge":
            num_add = 2
        elif processing_type == "ovl":
            num_add = 2

        # Select random add images
        selected_add_images = random.sample(self.add_images, min(num_add, len(self.add_images)))
        
        # Copy of original bounding boxes
        augmented_bboxes = bboxes.copy()
        augmented_image = image.copy()
        
        # For each selected add image
        for add_image_name in selected_add_images:
            # Load the add image
            add_image_path = os.path.join(self.add_images_folder, add_image_name)
            try:
                add_image = cv2.imread(add_image_path, cv2.IMREAD_UNCHANGED)
                if add_image is None:
                    logger.warning(f"Failed to load add image: {add_image_path}")
                    continue
            except Exception as e:
                logger.warning(f"Error loading add image {add_image_path}: {e}")
                continue
            
            # Ensure add image has alpha channel (if it's a 3-channel image)
            if add_image.shape[2] == 3:
                add_image = cv2.cvtColor(add_image, cv2.COLOR_BGR2BGRA)
                add_image[:, :, 3] = 255  # Set full opacity
            
            # Determine augmentation parameters
            angle = random.uniform(self.min_rotate, self.max_rotate)
            
            # Calculate appropriate zoom based on largest bounding box with class_id=1
            add_height, add_width = add_image.shape[:2]
            
            
            # Place the object
            augmented_image, new_bbox = self._place_object(augmented_image, add_image, augmented_bboxes, angle, target_height, target_width, processing_type)
            
            # If placement successful, add the new bounding box
            if new_bbox is not None:
                if processing_type == "edge":
                    count_edge += 1
                elif processing_type == "ovl":
                    count_ovl += 1
                else:
                    count_none += 1
                # if bbox got out of image, resize it to fit in image
                if not new_bbox.is_inside_image():
                    old_top_left = (new_bbox.x_center - new_bbox.width / 2, new_bbox.y_center - new_bbox.height / 2)
                    old_bottom_right = (new_bbox.x_center + new_bbox.width / 2, new_bbox.y_center + new_bbox.height / 2)
                    new_top_left = (max(0, old_top_left[0]), max(0, old_top_left[1]))
                    new_bottom_right = (min(1, old_bottom_right[0]), min(1, old_bottom_right[1]))
                    new_width = new_bottom_right[0] - new_top_left[0]
                    new_height = new_bottom_right[1] - new_top_left[1]
                    new_x_center = (new_top_left[0] + new_bottom_right[0]) / 2
                    new_y_center = (new_top_left[1] + new_bottom_right[1]) / 2
                    new_bbox = BoundingBox(new_bbox.class_id, new_x_center, new_y_center, new_width, new_height)
                augmented_bboxes.append(new_bbox)
        
        # Save the augmented image and labels
        # encode
        image_name = os.path.basename(image_path)
        label_name = os.path.basename(label_path)
        
        image_name = processing_type + "_" + f"{i:04d}_" + image_name
        label_name = processing_type + "_" + f"{i:04d}_" + label_name
        
        output_image_path = os.path.join(self.output_folder, "images", image_name)
        output_label_path = os.path.join(self.output_folder, "labels", label_name)
        
        try:
            cv2.imwrite(output_image_path, augmented_image)
            self._save_label(output_label_path, augmented_bboxes)
            logger.info(f"Successfully augmented {image_name} with {len(augmented_bboxes) - len(bboxes)} new objects")
            return output_image_path, output_label_path
        except Exception as e:
            logger.error(f"Error saving augmented data for {image_name}: {e}")
            return None, None
    
    def process_all(self, processing_type: str) -> None:
        """Process all images in the images folder."""
        # Get all image files
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in os.listdir(self.images_folder) 
                      if os.path.isfile(os.path.join(self.images_folder, f)) and 
                      os.path.splitext(f)[1].lower() in valid_extensions]
        # shuffle the image files
        random.shuffle(image_files)
        if not image_files:
            logger.error(f"No images found in {self.images_folder}")
            return
        
        logger.info(f"Found {len(image_files)} images to process")
        
        ratio = 0
        if processing_type == "none":
            ratio = NONE_RATIO
        elif processing_type == "edge":
            ratio = EDGE_RATIO
        elif processing_type == "ovl":
            ratio = OVL_RATIO
        else:
            logger.error(f"Invalid processing type: {processing_type}")
            return
        num_processed_images = int(len(image_files) * ratio)
        image_files = image_files[:num_processed_images]
        for i in range(len(image_files)):
            idx = i % len(image_files)
            image_file = image_files[idx]
            # Get the corresponding label file
            label_file = os.path.splitext(image_file)[0] + ".txt"
            label_path = os.path.join(self.labels_folder, label_file)
            
            # Skip if label file doesn't exist
            if not os.path.isfile(label_path):
                logger.warning(f"Label file not found for {image_file}, skipping")
                continue
            
            # Augment the image
            image_path = os.path.join(self.images_folder, image_file)
            self.augment_image(image_path, label_path, i, processing_type)
        
        logger.info("Processing complete")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Image Augmentation Tool")
    parser.add_argument("--images-folder", type=str, default=data_path + "/images",
                        help="Path to the folder containing original images")
    parser.add_argument("--labels-folder", type=str, default=data_path + "/labels",
                        help="Path to the folder containing original labels")
    parser.add_argument("--add-images-folder", type=str, default="add_images", 
                        help="Path to the folder containing images to add")
    parser.add_argument("--output-folder", type=str, default=output_dir, 
                        help="Path to the folder where augmented images and labels will be saved")
    parser.add_argument("--min-rotate", type=float, default=-30.0, 
                        help="Minimum rotation angle in degrees")
    parser.add_argument("--max-rotate", type=float, default=30.0, 
                        help="Maximum rotation angle in degrees")
    parser.add_argument("--min-zoom", type=float, default=0.5, 
                        help="Minimum zoom factor")
    parser.add_argument("--max-zoom", type=float, default=1.5, 
                        help="Maximum zoom factor")
    parser.add_argument("--full-ratio", type=float, default=0.3, 
                        help="Probability that the added image must be fully inside")
    parser.add_argument("--min-visibility", type=float, default=0.7, 
                        help="Minimum required visibility ratio")
    parser.add_argument("--max-attempts", type=int, default=10000, 
                        help="Maximum number of attempts to place an object")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    try:
        augmenter = ImageAugmenter(
            images_folder=args.images_folder,
            labels_folder=args.labels_folder,
            add_images_folder=args.add_images_folder,
            output_folder=args.output_folder,
            min_rotate=args.min_rotate,
            max_rotate=args.max_rotate,
            min_zoom=args.min_zoom,
            max_zoom=args.max_zoom,
            min_visibility=args.min_visibility,
            max_attempts=args.max_attempts
        )
        
        processing_type = "none"
        augmenter.process_all(processing_type)
        processing_type = "ovl"
        augmenter.process_all(processing_type)
        processing_type = "edge"
        augmenter.process_all(processing_type)
    
    
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 1
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return 1
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return 1
    print("count_none", count_none)
    print("count_edge", count_edge)
    print("count_ovl", count_ovl)
    return 0


if __name__ == "__main__":
    exit(main())