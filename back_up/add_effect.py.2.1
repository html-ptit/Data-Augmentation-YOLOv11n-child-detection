#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image Augmentation Tool
-----------------------
This script applies various image augmentation techniques to images in a given folder
and saves the results to a new folder.

Augmentation techniques:
- Blur
- Noise addition
- Histogram equalization
- Contrast stretching
- Striping effect (horizontal/vertical)
- Image blending
- Spatial filtering
- Thresholding
- Color matching
- Color grading

The script also saves information about applied effects to track augmentation history.
"""

import os
import random
import logging
import json
import datetime
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import yaml


def load_config(config_path='cfg_add_effect.yaml'):
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

# Access ratios
BLUR_RATIO = config['ratios']['blur']
NOISE_RATIO = config['ratios']['noise']
HISTOGRAM_EQUALIZATION_RATIO = config['ratios']['histogram_equalization']
CONTRAST_STRETCHING_RATIO = config['ratios']['contrast_stretching']
BLEND_RATIO = config['ratios']['blend']
STRIPES_RATIO = config['ratios']['stripes']
SPATIAL_FILTER_RATIO = config['ratios']['spatial_filter']
THRESHOLDING_RATIO = config['ratios']['thresholding']
COLOR_MATCHING_RATIO = config['ratios']['color_matching']
COLOR_GRADING_RATIO = config['ratios']['color_grading']

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define parameters boundaries
PARAMS = {
    'blur': {'min': config['params']['blur']['min'], 'max': config['params']['blur']['max']},  # Blur kernel size
    'noise': {'min': config['params']['noise']['min'], 'max': config['params']['noise']['max']},  # Noise strength
    'contrast': {'min': config['params']['contrast']['min'], 'max': config['params']['contrast']['max']},  # Contrast factor
    'stripes': {
        'min_width': config['params']['stripes']['min_width'], 'max_width': config['params']['stripes']['max_width'],  # Stripe width
        'min_opacity': config['params']['stripes']['min_opacity'], 'max_opacity': config['params']['stripes']['max_opacity'],  # Stripe opacity
        'min_count': config['params']['stripes']['min_count'], 'max_count': config['params']['stripes']['max_count']  # Number of stripes
    },
    'blend': {'min': config['params']['blend']['min'], 'max': config['params']['blend']['max']},  # Blend factor
    'spatial_filter': {'min': config['params']['spatial_filter']['min'], 'max': config['params']['spatial_filter']['max']},  # Filter kernel size
    'threshold': {'min': config['params']['threshold']['min'], 'max': config['params']['threshold']['max']},  # Threshold value
    'color_matching': {
        'min_strength': config['params']['color_matching']['min_strength'],
        'max_strength': config['params']['color_matching']['max_strength'],  # Color matching strength
        # Reference colors for color matching
        'reference_colors': [
            config['params']['color_matching']['reference_colors'][0],
            config['params']['color_matching']['reference_colors'][1],
            config['params']['color_matching']['reference_colors'][2],
            config['params']['color_matching']['reference_colors'][3],
            config['params']['color_matching']['reference_colors'][4]
        ]
    },
    'color_grading': {
        'shadows': {'min': config['params']['color_grading']['shadows']['min'], 'max': config['params']['color_grading']['shadows']['max']},  # Adjustment for shadows
        'midtones': {'min': config['params']['color_grading']['midtones']['min'], 'max': config['params']['color_grading']['midtones']['max']},  # Adjustment for midtones
        'highlights': {'min': config['params']['color_grading']['highlights']['min'], 'max': config['params']['color_grading']['highlights']['max']},  # Adjustment for highlights
        'saturation': {'min': config['params']['color_grading']['saturation']['min'], 'max': config['params']['color_grading']['saturation']['max']}  # Saturation adjustment
    }
}

class ImageAugmenter:
    """Class for image augmentation operations."""
    
    def __init__(self, input_dir, output_dir, effects_dir="effects"):
        """
        Initialize the ImageAugmenter.
        
        Args:
            input_dir (str): Directory containing images to process
            output_dir (str): Directory to save processed images
            effects_dir (str): Directory to save effects information
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.effects_dir = Path(effects_dir)
        
        # Create output and effects directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.effects_dir, exist_ok=True)
        
        # Get list of image files
        self.image_files = [f for f in self.input_dir.glob("*") 
                           if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        logger.info(f"Found {len(self.image_files)} images in {self.input_dir}")
    
    def apply_blur(self, image, strength=None):
        """
        Apply Gaussian blur to an image.
        
        Args:
            image (numpy.array): Input image
            strength (int, optional): Blur kernel size. If None, a random value is chosen.
            
        Returns:
            numpy.array: Blurred image
        """
        if strength is None:
            # Make sure kernel size is odd
            strength = random.randrange(PARAMS['blur']['min'], PARAMS['blur']['max'] + 1, 2)
        
        return cv2.GaussianBlur(image, (strength, strength), 0)
    
    def apply_noise(self, image, strength=None):
        """
        Add Gaussian noise to an image.
        
        Args:
            image (numpy.array): Input image
            strength (float, optional): Noise standard deviation. If None, a random value is chosen.
            
        Returns:
            numpy.array: Noisy image
        """
        if strength is None:
            strength = random.uniform(PARAMS['noise']['min'], PARAMS['noise']['max'])
        
        # Generate Gaussian noise
        noise = np.random.normal(0, strength, image.shape).astype(np.int16)
        
        # Add noise to the image
        noisy_image = cv2.add(image.astype(np.int16), noise)
        
        # Clip to ensure valid pixel values
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image
    
    def apply_histogram_equalization(self, image):
        """
        Apply histogram equalization to enhance contrast.
        
        Args:
            image (numpy.array): Input image
            
        Returns:
            numpy.array: Equalized image
        """
        # Check if image is grayscale or color
        if len(image.shape) == 2 or image.shape[2] == 1:
            return cv2.equalizeHist(image)
        else:
            # Convert to YUV and equalize Y channel only
            img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    def apply_contrast_stretching(self, image, strength=None):
        """
        Apply contrast stretching to an image.
        
        Args:
            image (numpy.array): Input image
            strength (float, optional): Contrast factor. If None, a random value is chosen.
            
        Returns:
            numpy.array: Image with adjusted contrast
        """
        if strength is None:
            strength = random.uniform(PARAMS['contrast']['min'], PARAMS['contrast']['max'])
        
        # Convert to float for scaling
        img_float = image.astype(np.float32) / 255.0
        
        # Apply contrast stretching
        # Mean centering and scaling
        mean = np.mean(img_float, axis=(0, 1), keepdims=True)
        adjusted = (img_float - mean) * strength + mean
        
        # Clip and convert back to uint8
        adjusted = np.clip(adjusted * 255, 0, 255).astype(np.uint8)
        
        return adjusted
    
    def apply_stripes(self, image, direction=None, params=None):
        """
        Add striping effect (horizontal or vertical) to simulate camera interference.
        
        Args:
            image (numpy.array): Input image
            direction (str, optional): 'horizontal' or 'vertical'. If None, randomly chosen.
            params (dict, optional): Dictionary with 'width', 'opacity', and 'count' keys.
                                    If None, random values are chosen.
            
        Returns:
            numpy.array: Image with stripe effect
        """
        if direction is None:
            direction = random.choice(['horizontal', 'vertical'])
        
        if params is None:
            params = {
                'width': random.randint(PARAMS['stripes']['min_width'], PARAMS['stripes']['max_width']),
                'opacity': random.uniform(PARAMS['stripes']['min_opacity'], PARAMS['stripes']['max_opacity']),
                'count': random.randint(PARAMS['stripes']['min_count'], PARAMS['stripes']['max_count'])
            }
        
        # Create a copy of the image
        result = image.copy()
        height, width = image.shape[:2]
        
        # Generate stripe positions
        if direction == 'horizontal':
            positions = np.random.choice(range(height), params['count'], replace=False)
            for pos in positions:
                start = max(0, pos - params['width'] // 2)
                end = min(height, pos + params['width'] // 2 + 1)
                # Apply a semi-transparent white strip
                # Ensure same data type for overlay and source
                overlay = np.ones_like(result[start:end, :]) * 255
                result[start:end, :] = cv2.addWeighted(
                    result[start:end, :].astype(np.float32), 1 - params['opacity'], 
                    overlay.astype(np.float32), params['opacity'], 0
                ).astype(np.uint8)
        else:  # vertical
            positions = np.random.choice(range(width), params['count'], replace=False)
            for pos in positions:
                start = max(0, pos - params['width'] // 2)
                end = min(width, pos + params['width'] // 2 + 1)
                # Apply a semi-transparent white strip
                # Ensure same data type for overlay and source
                overlay = np.ones_like(result[:, start:end]) * 255
                result[:, start:end] = cv2.addWeighted(
                    result[:, start:end].astype(np.float32), 1 - params['opacity'], 
                    overlay.astype(np.float32), params['opacity'], 0
                ).astype(np.uint8)
        
        return result
    
    def apply_blending(self, image, strength=None):
        """
        Blend the image with a random pattern for texture effect.
        
        Args:
            image (numpy.array): Input image
            strength (float, optional): Blending factor. If None, a random value is chosen.
            
        Returns:
            numpy.array: Blended image
        """
        if strength is None:
            strength = random.uniform(PARAMS['blend']['min'], PARAMS['blend']['max'])
        
        # Create a random texture of the same size as the image
        texture = np.random.randint(0, 256, image.shape, dtype=np.uint8)
        
        # Blend the image with the texture
        blended = cv2.addWeighted(
            image.astype(np.float32), 1 - strength, 
            texture.astype(np.float32), strength, 
            0
        ).astype(np.uint8)
        
        return blended
    
    def apply_spatial_filter(self, image, kernel_size=None):
        """
        Apply a spatial filter (median filter in this case).
        
        Args:
            image (numpy.array): Input image
            kernel_size (int, optional): Size of the median filter kernel.
                                        If None, a random value is chosen.
            
        Returns:
            numpy.array: Filtered image
        """
        if kernel_size is None:
            # Make sure kernel size is odd
            kernel_size = random.randrange(PARAMS['spatial_filter']['min'], 
                                          PARAMS['spatial_filter']['max'] + 1, 2)
        
        return cv2.medianBlur(image, kernel_size)
    
    def apply_thresholding(self, image, threshold=None):
        """
        Apply thresholding to an image.
        
        Args:
            image (numpy.array): Input image
            threshold (int, optional): Threshold value. If None, a random value is chosen.
            
        Returns:
            numpy.array: Thresholded image
        """
        if threshold is None:
            threshold = random.randint(PARAMS['threshold']['min'], PARAMS['threshold']['max'])
        
        # Convert to grayscale if it's a color image
        if len(image.shape) > 2 and image.shape[2] > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply thresholding
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # If original image was color, convert thresholded image back to color
        if len(image.shape) > 2 and image.shape[2] > 1:
            thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            
        return thresh
        
    def apply_color_matching(self, image, strength=None, reference_color=None):
        """
        Apply color matching effect to shift image colors towards a reference color.
        
        Args:
            image (numpy.array): Input image
            strength (float, optional): Strength of color matching effect.
                                        If None, a random value is chosen.
            reference_color (list, optional): RGB color to match.
                                            If None, a random color is chosen.
            
        Returns:
            numpy.array: Color matched image
        """
        if strength is None:
            strength = random.uniform(
                PARAMS['color_matching']['min_strength'], 
                PARAMS['color_matching']['max_strength']
            )
            
        if reference_color is None:
            reference_color = random.choice(PARAMS['color_matching']['reference_colors'])
        
        # Convert RGB to BGR (OpenCV format)
        reference_color = np.array(reference_color[::-1], dtype=np.float32).reshape(1, 1, 3)
        
        # Create overlay with broadcast
        overlay = np.ones_like(image, dtype=np.float32) * reference_color

        # Convert image to float32
        image_float = image.astype(np.float32)

        # Blend images
        matched_image = cv2.addWeighted(
            image_float, 1 - strength,
            overlay, strength,
            0
        ).astype(np.uint8)

        return matched_image
        
    def apply_color_grading(self, image, params=None):
        """
        Apply color grading to adjust shadows, midtones, highlights and saturation.
        
        Args:
            image (numpy.array): Input image
            params (dict, optional): Color grading parameters.
                                    If None, random values are chosen.
            
        Returns:
            numpy.array: Color graded image
        """
        if params is None:
            params = {
                'shadows': random.randint(
                    PARAMS['color_grading']['shadows']['min'],
                    PARAMS['color_grading']['shadows']['max']
                ),
                'midtones': random.randint(
                    PARAMS['color_grading']['midtones']['min'],
                    PARAMS['color_grading']['midtones']['max']
                ),
                'highlights': random.randint(
                    PARAMS['color_grading']['highlights']['min'],
                    PARAMS['color_grading']['highlights']['max']
                ),
                'saturation': random.uniform(
                    PARAMS['color_grading']['saturation']['min'],
                    PARAMS['color_grading']['saturation']['max']
                )
            }
        
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Split channels
        b, g, r = cv2.split(img_float)
        
        # Convert to HSV for saturation adjustment
        hsv = cv2.cvtColor(img_float, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Adjust shadows (dark pixels)
        shadow_mask = v < 0.3
        v[shadow_mask] = np.clip(v[shadow_mask] + params['shadows'] / 255.0, 0, 1)
        
        # Adjust midtones (medium pixels)
        midtone_mask = (v >= 0.3) & (v <= 0.7)
        v[midtone_mask] = np.clip(v[midtone_mask] + params['midtones'] / 255.0, 0, 1)
        
        # Adjust highlights (bright pixels)
        highlight_mask = v > 0.7
        v[highlight_mask] = np.clip(v[highlight_mask] + params['highlights'] / 255.0, 0, 1)
        
        # Adjust saturation
        s = np.clip(s * params['saturation'], 0, 1)
        
        # Merge channels back
        hsv_merged = cv2.merge([h, s, v])
        graded_image = cv2.cvtColor(hsv_merged, cv2.COLOR_HSV2BGR)
        
        # Convert back to uint8
        graded_image = np.clip(graded_image * 255, 0, 255).astype(np.uint8)
        
        return graded_image
    
    def process_image(self, image_path):
        """
        Process a single image with random augmentations.
        
        Args:
            image_path (Path): Path to the input image file
            
        Returns:
            tuple: (processed_image, effects_info) where processed_image is the augmented 
                  image and effects_info is a dictionary with information about applied effects
        """
        try:
            # Read the image
            image = cv2.imread(str(image_path))
            
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return None, None
                
            # Randomly select and apply augmentations
            # Create a list of all augmentation methods with their random application probability
            augmentations = [
                (self.apply_blur, BLUR_RATIO),
                (self.apply_noise, NOISE_RATIO),
                (self.apply_histogram_equalization, HISTOGRAM_EQUALIZATION_RATIO),
                (self.apply_contrast_stretching, CONTRAST_STRETCHING_RATIO),
                (self.apply_stripes, STRIPES_RATIO),
                (self.apply_blending, BLEND_RATIO),
                (self.apply_spatial_filter, SPATIAL_FILTER_RATIO),
                (self.apply_thresholding, THRESHOLDING_RATIO),
                (self.apply_color_matching, COLOR_MATCHING_RATIO),
                (self.apply_color_grading, COLOR_GRADING_RATIO)
            ]
            
            # Process with selected augmentations
            processed_image = image.copy()
            effects_info = {
                "image_name": image_path.name,
                "processing_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "original_size": f"{image.shape[1]}x{image.shape[0]}",
                "applied_effects": []
            }
            
            for augmentation_func, probability in augmentations:
                if random.random() < probability:
                    # Get the function name without the 'apply_' prefix
                    effect_name = augmentation_func.__name__.replace('apply_', '')
                    
                    # Initialize effect parameters dictionary
                    effect_params = {}
                    
                    # Apply different effects with appropriate parameters
                    if effect_name == 'blur':
                        kernel_size = random.randrange(PARAMS['blur']['min'], PARAMS['blur']['max'] + 1, 2)
                        processed_image = augmentation_func(processed_image, kernel_size)
                        effect_params['kernel_size'] = kernel_size
                        
                    elif effect_name == 'noise':
                        strength = random.uniform(PARAMS['noise']['min'], PARAMS['noise']['max'])
                        processed_image = augmentation_func(processed_image, strength)
                        effect_params['strength'] = round(strength, 2)
                        
                    elif effect_name == 'histogram_equalization':
                        processed_image = augmentation_func(processed_image)
                        
                    elif effect_name == 'contrast_stretching':
                        strength = random.uniform(PARAMS['contrast']['min'], PARAMS['contrast']['max'])
                        processed_image = augmentation_func(processed_image, strength)
                        effect_params['strength'] = round(strength, 2)
                        
                    elif effect_name == 'stripes':
                        direction = random.choice(['horizontal', 'vertical'])
                        params = {
                            'width': random.randint(PARAMS['stripes']['min_width'], PARAMS['stripes']['max_width']),
                            'opacity': random.uniform(PARAMS['stripes']['min_opacity'], PARAMS['stripes']['max_opacity']),
                            'count': random.randint(PARAMS['stripes']['min_count'], PARAMS['stripes']['max_count'])
                        }
                        processed_image = augmentation_func(processed_image, direction, params)
                        effect_params.update({
                            'direction': direction,
                            'width': params['width'],
                            'opacity': round(params['opacity'], 2),
                            'count': params['count']
                        })
                        
                    elif effect_name == 'blending':
                        strength = random.uniform(PARAMS['blend']['min'], PARAMS['blend']['max'])
                        processed_image = augmentation_func(processed_image, strength)
                        effect_params['strength'] = round(strength, 2)
                        
                    elif effect_name == 'spatial_filter':
                        kernel_size = random.randrange(PARAMS['spatial_filter']['min'], 
                                                     PARAMS['spatial_filter']['max'] + 1, 2)
                        processed_image = augmentation_func(processed_image, kernel_size)
                        effect_params['kernel_size'] = kernel_size
                        
                    elif effect_name == 'thresholding':
                        threshold = random.randint(PARAMS['threshold']['min'], PARAMS['threshold']['max'])
                        processed_image = augmentation_func(processed_image, threshold)
                        effect_params['threshold'] = threshold
                        
                    elif effect_name == 'color_matching':
                        strength = random.uniform(
                            PARAMS['color_matching']['min_strength'], 
                            PARAMS['color_matching']['max_strength']
                        )
                        ref_color = random.choice(PARAMS['color_matching']['reference_colors'])
                        processed_image = augmentation_func(processed_image, strength, ref_color)
                        effect_params.update({
                            'strength': round(strength, 2),
                            'reference_color': ref_color
                        })
                        
                    elif effect_name == 'color_grading':
                        params = {
                            'shadows': random.randint(
                                PARAMS['color_grading']['shadows']['min'],
                                PARAMS['color_grading']['shadows']['max']
                            ),
                            'midtones': random.randint(
                                PARAMS['color_grading']['midtones']['min'],
                                PARAMS['color_grading']['midtones']['max']
                            ),
                            'highlights': random.randint(
                                PARAMS['color_grading']['highlights']['min'],
                                PARAMS['color_grading']['highlights']['max']
                            ),
                            'saturation': random.uniform(
                                PARAMS['color_grading']['saturation']['min'],
                                PARAMS['color_grading']['saturation']['max']
                            )
                        }
                        processed_image = augmentation_func(processed_image, params)
                        effect_params.update({
                            'shadows': params['shadows'],
                            'midtones': params['midtones'],
                            'highlights': params['highlights'],
                            'saturation': round(params['saturation'], 2)
                        })
                    
                    # Add effect info to the dictionary
                    effects_info['applied_effects'].append({
                        'name': effect_name,
                        'parameters': effect_params
                    })
            
            if effects_info['applied_effects']:
                effect_names = [effect['name'] for effect in effects_info['applied_effects']]
                logger.info(f"Applied {', '.join(effect_names)} to {image_path.name}")
            else:
                logger.info(f"No augmentations were applied to {image_path.name}")
                
            return processed_image, effects_info
            
        except Exception as e:
            logger.error(f"Error processing image {image_path.name}: {str(e)}")
            return None, None
    
    def process_all_images(self):
        """Process all images in the input directory and save to output directory."""
        logger.info(f"Starting processing of {len(self.image_files)} images")
        
        for image_path in tqdm(self.image_files, desc="Processing images"):
            output_path = self.output_dir / image_path.name
            effects_path = self.effects_dir / f"{image_path.stem}.json"
            
            # Process the image
            processed_image, effects_info = self.process_image(image_path)
            
            if processed_image is not None and effects_info is not None:
                # Save the processed image
                try:
                    cv2.imwrite(str(output_path), processed_image)
                    logger.debug(f"Saved processed image to {output_path}")
                    
                    # Save effects information
                    with open(effects_path, 'w', encoding='utf-8') as f:
                        json.dump(effects_info, f, indent=2)
                    logger.debug(f"Saved effects information to {effects_path}")
                    
                except Exception as e:
                    logger.error(f"Error saving outputs for {image_path.name}: {str(e)}")
        
        logger.info(f"Finished processing. Images saved to {self.output_dir}")
        logger.info(f"Effects information saved to {self.effects_dir}")


def main():
    """Main function to run the image augmentation process."""
    input_dir = "images"
    output_dir = "new_images"
    effects_dir = "effects"
    
    # Verify input directory exists
    if not os.path.isdir(input_dir):
        logger.error(f"Input directory '{input_dir}' does not exist.")
        return
    
    # Create instance of ImageAugmenter and process images
    augmenter = ImageAugmenter(input_dir, output_dir, effects_dir)
    augmenter.process_all_images()


if __name__ == "__main__":
    main()