# Image Augmentation Toolkit

Two complementary Python tools for comprehensive image augmentation to enhance computer vision datasets.

## Overview

This toolkit contains two specialized augmentation scripts:

### 1. Object Addition Tool (`augment_images.py`)
Augments object detection datasets by adding additional objects with proper bounding box handling.

### 2. Image Effects Tool (First script in document)
Applies various visual effects and distortions to simulate real-world image conditions.

## Object Addition Tool Features

- **Smart Object Placement**: Automatically places objects with collision detection and visibility constraints
- **YOLO Format Support**: Handles YOLO format bounding box annotations seamlessly  
- **Background-Aware Blending**: Dynamic alpha blending based on background complexity
- **Size-Aware Scaling**: Automatically scales added objects based on existing objects in the scene

## Image Effects Tool Features

- **Visual Effects**: Blur, noise, histogram equalization, contrast stretching
- **Distortion Effects**: Striping (horizontal/vertical), spatial filtering, thresholding
- **Color Effects**: Color matching, color grading, image blending
- **Effect Tracking**: Saves detailed information about applied effects to JSON files

## Quick Start

### Object Addition Tool
```bash
python add_image.py \
    --images-folder "images" \
    --labels-folder "labels" \
    --add-images-folder "add_images" \
    --output-folder "new_data"
```

### Image Effects Tool
```bash
python add_effect.py  # Uses 'images' folder by default
```

## Configuration

Both tools use YAML configuration files:
- Object Addition: `cfg_add_image.yaml`
- Image Effects: `cfg_add_effect.yaml`

Configure ratios (0.0-1.0) for each effect type and parameter ranges.


## Dependencies

```bash
pip install opencv-python numpy PyYAML tqdm matplotlib
```

Based on your folder structure and explanation, here's how your data organization works:

## Folder Structure Breakdown

**`add_images/`** - Source images for object insertion
- Contains child images with backgrounds already removed (PNG with transparency)
- These are the "clean" objects ready to be added to scenes
- Used by `add_image.py` as the source material

**`new_images/`** - Raw child images  
- Original photos of children before background removal
- These need preprocessing to remove backgrounds before use
- Not directly used by the augmentation tools

**`dataset/`** - Training data organized by augmentation type:

**`dataset/add_child/`** - Object addition results
- `images/` - Base scenes where child objects will be added
- `labels/` - YOLO format bounding box annotations for existing objects in scenes

**`dataset/add_effect/`** - Image effects results  
- `images/` - Images to apply visual effects/distortions to
- `labels/` - YOLO annotations (maintained through effect processing)

## Workflow

1. **Prepare objects**: Process images in `new_images/` to remove backgrounds → save to `add_images/`
2. **Object addition**: Use `add_image.py` to add objects from `add_images/` to scenes in `dataset/add_child/images/`
3. **Apply effects**: Use `add_effect.py` to add visual distortions to images in `dataset/add_effect/images/`

The separation allows you to apply different augmentation strategies to different subsets of your training data, with proper annotation handling throughout the pipeline.