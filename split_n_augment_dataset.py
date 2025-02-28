#!/usr/bin/env python3
"""
Surgical Instrument Dataset Augmenter

This script provides functionality for:
1. Augmenting images of surgical instruments with various transformations
2. Maintaining accurate annotations (bounding boxes and polygons) during augmentation
3. Splitting datasets into train/validation/test sets

Each transformation preserves the spatial relationship between the image and its annotations,
allowing for the creation of a larger, more diverse training dataset while maintaining
label accuracy for both object detection and segmentation tasks.
"""

import os
import json
import cv2
import numpy as np
import albumentations as A
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import argparse
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import random  # Needed for split_dataset
import shutil  # Needed for file copying in split_dataset
import config  # Needed for configuration

class InstrumentDatasetAugmenter:
    """
    Handles augmentation of surgical instrument dataset, maintaining both 
    bounding box and polygon annotations throughout transformations.
    
    This augmenter applies multiple image transformations (brightness, rotation, flips, etc.)
    to input images while properly transforming their corresponding annotations.
    """
    def __init__(
        self, 
        image_dir: str, 
        json_dir: str,
        output_image_dir: str,
        output_json_dir: str,
        max_size: int = 1024
    ):
        """
        Initialize the augmenter with input/output directories and settings.
        
        Args:
            image_dir: Directory containing original images
            json_dir: Directory containing original JSON annotations
            output_image_dir: Directory to save augmented images
            output_json_dir: Directory to save augmented annotations
            max_size: Maximum dimension for resizing images (maintains aspect ratio)
        """
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.output_image_dir = output_image_dir
        self.output_json_dir = output_json_dir
        self.max_size = max_size
        
        # Create output directories if they don't exist
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_json_dir, exist_ok=True)
        
        # Define resize transform
        self.resize_transform = A.Compose([
            A.LongestMaxSize(max_size=max_size, p=1)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
           keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels']))
        
        # Define individual augmentations for separate application
        self.augmentations = {
            'brightness': A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1)
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
               keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels'])),
            
            'rotate': A.Compose([
                A.Rotate(limit=30, p=1)
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
               keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels'])),
            
            'hflip': A.Compose([
                A.HorizontalFlip(p=1)
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
               keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels'])),
            
            'vflip': A.Compose([
                A.VerticalFlip(p=1)
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
               keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels'])),
            
            'hsv': A.Compose([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1)
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
               keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels'])),
            
            'blur': A.Compose([
                A.Blur(blur_limit=5, p=1)
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
               keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels'])),
            
            'noise': A.Compose([
                A.GaussNoise(std_range=(0.1, 0.2), p=1)
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
               keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels'])),
            
            'sharpen': A.Compose([
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1)
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
               keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels'])),
            
            'rgbshift': A.Compose([
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1)
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
               keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels'])),
            
            'clahe': A.Compose([
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1)
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
               keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels']))
        }

    def get_json_path(self, image_path: str) -> str:
        """
        Convert image path to corresponding JSON annotation path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Path to the corresponding JSON annotation file
        """
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(self.json_dir, f"{base_name}.json")

    def load_annotation(self, json_path: str) -> Dict:
        """
        Load annotation file from JSON.
        
        Args:
            json_path: Path to the JSON annotation file
            
        Returns:
            Dictionary containing annotations, or empty dictionary if file not found
        """
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Annotation file not found: {json_path}")
            return {"instruments": []}

    def prepare_augmentation_inputs(self, image: np.ndarray, annotation: Dict) -> Tuple:
        """
        Prepare bounding boxes and keypoints for Albumentations transformations.
        
        Converts our annotation format to the format expected by Albumentations.
        
        Args:
            image: Input image as numpy array
            annotation: Dictionary containing annotation data
            
        Returns:
            Tuple of (bboxes, bbox_labels, keypoints, keypoint_labels)
            - bboxes: List of bounding boxes in Pascal VOC format [x1, y1, x2, y2]
            - bbox_labels: List of labels for each bounding box
            - keypoints: List of keypoints [(x, y), ...] from polygon vertices
            - keypoint_labels: List of labels for each keypoint
        """
        bboxes = []
        bbox_labels = []
        keypoints = []
        keypoint_labels = []
        
        for instrument in annotation['instruments']:
            # Add bbox
            x, y, w, h = instrument['bbox']
            bboxes.append([x, y, x+w, y+h])  # Convert to pascal_voc format
            bbox_labels.append(instrument['name'])
            
            # Add polygon points as keypoints
            polygon = instrument['polygon']
            keypoints.extend([(x, y) for x, y in polygon])
            keypoint_labels.extend([instrument['name']] * len(polygon))
            
        return bboxes, bbox_labels, keypoints, keypoint_labels

    def reconstruct_annotation(self, 
                             transformed: Dict, 
                             original_annotation: Dict) -> Dict:
        """
        Reconstruct annotation format from transformed data.
        
        Converts data from Albumentations format back to our annotation format.
        
        Args:
            transformed: Dictionary with transformed data from Albumentations
            original_annotation: Original annotation dictionary for structure reference
            
        Returns:
            Reconstructed annotation dictionary with updated coordinates
        """
        new_annotation = original_annotation.copy()
        new_annotation['instruments'] = []
        
        # Group keypoints by label to reconstruct polygons
        keypoint_groups = {}
        for idx, (kp, label) in enumerate(zip(transformed['keypoints'], 
                                            transformed['keypoint_labels'])):
            if label not in keypoint_groups:
                keypoint_groups[label] = []
            keypoint_groups[label].append(kp)
            
        # Reconstruct instruments
        for bbox, bbox_label in zip(transformed['bboxes'], 
                                  transformed['bbox_labels']):
            x1, y1, x2, y2 = bbox
            instrument = {
                'name': bbox_label,
                'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                'polygon': [[int(x), int(y)] for x, y in keypoint_groups[bbox_label]]
            }
            new_annotation['instruments'].append(instrument)
            
        return new_annotation

    def resize_image_and_annotations(self, image: np.ndarray, annotation: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Resize image while properly transforming annotations.
        
        Maintains aspect ratio and ensures annotations are correctly positioned.
        
        Args:
            image: Input image as numpy array
            annotation: Dictionary containing annotation data
            
        Returns:
            Tuple of (resized_image, resized_annotation)
        """
        # Prepare inputs for resize transform
        bboxes, bbox_labels, keypoints, keypoint_labels = self.prepare_augmentation_inputs(image, annotation)
        
        # Apply resize transform
        transformed = self.resize_transform(
            image=image,
            bboxes=bboxes,
            bbox_labels=bbox_labels,
            keypoints=keypoints,
            keypoint_labels=keypoint_labels
        )
        
        # Reconstruct annotation with resized coordinates
        resized_annotation = self.reconstruct_annotation(transformed, annotation)
        
        return transformed['image'], resized_annotation

    def visualize_augmentation(self, 
                             original_image: np.ndarray,
                             augmented_image: np.ndarray,
                             original_annotation: Dict,
                             augmented_annotation: Dict,
                             aug_name: str = ""):
        """
        Visualize original and augmented images with annotations.
        
        Creates a side-by-side plot showing the original and augmented images
        with their bounding boxes and polygons overlaid.
        
        Args:
            original_image: Original image as numpy array
            augmented_image: Augmented image as numpy array
            original_annotation: Original annotation dictionary
            augmented_annotation: Augmented annotation dictionary
            aug_name: Name of the augmentation for display purposes
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot original
        ax1.imshow(original_image)
        ax1.set_title(f'Original Image {original_image.shape}')
        self._draw_annotations(ax1, original_image, original_annotation)
        
        # Plot augmented
        ax2.imshow(augmented_image)
        ax2.set_title(f'Augmented Image ({aug_name}) {augmented_image.shape}')
        self._draw_annotations(ax2, augmented_image, augmented_annotation)
        
        plt.tight_layout()
        plt.show()
        
    def _draw_annotations(self, ax, image: np.ndarray, annotation: Dict):
        """
        Helper function to draw annotations on a matplotlib axis.
        
        Draws both polygons and bounding boxes with labels.
        
        Args:
            ax: Matplotlib axis to draw on
            image: The image being annotated
            annotation: Dictionary containing annotation data
        """
        for instrument in annotation['instruments']:
            # Draw polygon
            poly = np.array(instrument['polygon'])
            ax.plot(poly[:, 0], poly[:, 1], '-r', linewidth=2)
            
            # Draw bbox
            x, y, w, h = instrument['bbox']
            rect = plt.Rectangle((x, y), w, h, fill=False, color='g', linewidth=2)
            ax.add_patch(rect)
            
            # Add label
            ax.text(x, y-5, instrument['name'][:20], color='blue', fontsize=8)

    def verify_saved_augmentations(self, image_file: str):
        """
        Load and visualize all augmentations for a specific image.
        
        Useful for verifying that augmentations were saved correctly.
        
        Args:
            image_file: Filename of the image to verify
        """
        # Load original image and annotation
        image_path = os.path.join(self.image_dir, image_file)
        json_path = self.get_json_path(image_path)
        
        if not os.path.exists(image_path):
            print(f"Original image not found: {image_path}")
            return
            
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_annotation = self.load_annotation(json_path)
        
        # First show resize result
        resized_image, resized_annotation = self.resize_image_and_annotations(
            original_image, original_annotation)
        print("\nVerifying resize transform:")
        self.visualize_augmentation(
            original_image,
            resized_image,
            original_annotation,
            resized_annotation,
            "resize"
        )
        
        # Then show each augmentation
        for aug_name in self.augmentations.keys():
            base_name = os.path.splitext(image_file)[0]
            aug_image_path = os.path.join(
                self.output_image_dir, 
                f'{base_name}_{aug_name}.jpg'
            )
            aug_json_path = os.path.join(
                self.output_json_dir,
                f'{base_name}_{aug_name}.json'
            )
            
            if os.path.exists(aug_image_path) and os.path.exists(aug_json_path):
                aug_image = cv2.imread(aug_image_path)
                aug_image = cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB)
                aug_annotation = self.load_annotation(aug_json_path)
                
                print(f"\nVerifying {aug_name} augmentation:")
                self.visualize_augmentation(
                    resized_image, 
                    aug_image,
                    resized_annotation,
                    aug_annotation,
                    aug_name
                )

    def augment_dataset(self):
        """
        Augment entire dataset with individual augmentations.
        
        Main entry point for augmentation. Processes all images in the input
        directory, applies all defined augmentations, and saves the results.
        """
        image_files = [f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()
        
        for idx, image_file in enumerate(image_files):
            print(f"Processing image {idx}/{len(image_files)}: {image_file}")
            
            # Load image and annotation
            image_path = os.path.join(self.image_dir, image_file)
            json_path = self.get_json_path(image_path)
            
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
                
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            annotation = self.load_annotation(json_path)
            
            # First resize the image
            try:
                image, annotation = self.resize_image_and_annotations(image, annotation)
            except Exception as e:
                print(f"Error resizing {image_file}: {str(e)}")
                continue
            
            # Then apply each augmentation separately
            for aug_name, transform in self.augmentations.items():
                try:
                    # Prepare augmentation inputs
                    bboxes, bbox_labels, keypoints, keypoint_labels = \
                        self.prepare_augmentation_inputs(image, annotation)
                    
                    # Apply augmentation
                    transformed = transform(
                        image=image,
                        bboxes=bboxes,
                        bbox_labels=bbox_labels,
                        keypoints=keypoints,
                        keypoint_labels=keypoint_labels
                    )
                    
                    # Reconstruct annotation
                    augmented_annotation = self.reconstruct_annotation(transformed, annotation)
                    
                    # Save augmented data with augmentation name in filename
                    base_name = os.path.splitext(image_file)[0]
                    output_image_path = os.path.join(
                        self.output_image_dir, 
                        f'{base_name}_{aug_name}.jpg'
                    )
                    output_json_path = os.path.join(
                        self.output_json_dir,
                        f'{base_name}_{aug_name}.json'
                    )
                    
                    # Save image
                    cv2.imwrite(output_image_path, 
                               cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
                    
                    # Save annotation
                    with open(output_json_path, 'w') as f:
                        json.dump(augmented_annotation, f, indent=2)
                        
                except Exception as e:
                    print(f"Error applying {aug_name} augmentation to {image_file}: {str(e)}")
                    continue
            
            # Save the resized original image and annotation
            base_name = os.path.splitext(image_file)[0]
            output_image_path = os.path.join(
                self.output_image_dir, 
                f'{base_name}_resized.jpg'
            )
            output_json_path = os.path.join(
                self.output_json_dir,
                f'{base_name}_resized.json'
            )
            
            # Save resized image
            cv2.imwrite(output_image_path, 
                       cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # Save resized annotation
            with open(output_json_path, 'w') as f:
                json.dump(annotation, f, indent=2)
            
            # Only visualize the first image
            if idx == 0:
                print("\nVerifying augmentations for first image:")
                self.verify_saved_augmentations(image_file)
            else:
                # Show simple progress without visualizations
                if idx % 50 == 0:
                    print(f"Processed {idx}/{len(image_files)} images...")

    def debug_single_image(self, image_file: str):
        """
        Debug augmentations on a single image.
        
        Interactive debugging tool to apply and visualize each augmentation
        separately on a single image.
        
        Args:
            image_file: Filename of the image to debug
        """
        print(f"\nDebug Mode: Processing {image_file}")
        
        # Load image and annotation
        image_path = os.path.join(self.image_dir, image_file)
        json_path = self.get_json_path(image_path)
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return
            
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotation = self.load_annotation(json_path)
        
        # First show original image with annotations
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        self._draw_annotations(plt.gca(), image, annotation)
        plt.title("Original Image with Annotations")
        plt.show()
        
        # Show resized image
        resized_image, resized_annotation = self.resize_image_and_annotations(
            image, annotation)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(resized_image)
        self._draw_annotations(plt.gca(), resized_image, resized_annotation)
        plt.title("Resized Image with Annotations")
        plt.show()
        
        # Show each augmentation one by one
        for aug_name, transform in self.augmentations.items():
            print(f"\nApplying {aug_name} augmentation...")
            
            try:
                # Prepare augmentation inputs
                bboxes, bbox_labels, keypoints, keypoint_labels = \
                    self.prepare_augmentation_inputs(resized_image, resized_annotation)
                
                # Apply augmentation
                transformed = transform(
                    image=resized_image,
                    bboxes=bboxes,
                    bbox_labels=bbox_labels,
                    keypoints=keypoints,
                    keypoint_labels=keypoint_labels
                )
                
                # Reconstruct annotation
                augmented_annotation = self.reconstruct_annotation(
                    transformed, resized_annotation)
                
                # Visualize
                plt.figure(figsize=(10, 10))
                plt.imshow(transformed['image'])
                self._draw_annotations(plt.gca(), transformed['image'], 
                                    augmented_annotation)
                plt.title(f"{aug_name} Augmentation")
                plt.show()
                
                # Ask user to continue
                input("Press Enter to see next augmentation...")
                
            except Exception as e:
                print(f"Error applying {aug_name} augmentation: {str(e)}")
                continue

def split_dataset(source_dir: Path, split_config: Dict):
    """
    Split dataset into train, val, and test sets.
    
    Randomly divides the dataset according to specified ratios and
    copies files to appropriate directories.
    
    Args:
        source_dir: Directory containing the source images
        split_config: Configuration dictionary with split ratios
        
    Returns:
        Dictionary with split information
    """
    # Set random seed for reproducibility
    random.seed(split_config['seed'])

    # Get all image files and verify
    image_files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
    total_files = len(image_files)
    if total_files == 0:
        raise ValueError(f"No images found in {source_dir}")
    
    # Calculate split sizes
    train_size = int(total_files * split_config['train_ratio'])
    val_size = int(total_files * split_config['val_ratio'])
    test_size = int(total_files * split_config['test_ratio'])

    # Verify splits add up to total
    if train_size + val_size + test_size != total_files:
        # Adjust train_size to account for rounding
        train_size = total_files - (val_size + test_size)

    # Verify split ratios are reasonable
    assert train_size > 0, "Train split too small"
    assert val_size > 0, "Validation split too small"
    assert test_size > 0, "Test split too small"
    
    # Randomly shuffle files
    random.shuffle(image_files)
      
    # Split files
    splits = {
        'train': image_files[:train_size],
        'val': image_files[train_size:train_size + val_size],
        'test': image_files[train_size + val_size:]
    }
    
    # Verify split sizes
    for split_name, files in splits.items():
        split_size = len(files)
        expected_size = {'train': train_size, 'val': val_size, 'test': test_size}[split_name]
        assert split_size == expected_size, f"{split_name} split size mismatch: got {split_size}, expected {expected_size}"
    
    # Create directories and copy files
    for split_name, files in splits.items():
        split_dir = config.get_processed_data_path(split_name, 'images')
        json_dir = config.get_processed_data_path(split_name, 'jsons')
        
        split_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files with verification
        for file in tqdm(files, desc=f"Copying {split_name} files"):
            # Copy image
            shutil.copy2(file, split_dir / file.name)
            
            # Copy and verify JSON
            json_file = config.RAW_JSONS / f"{file.stem}.json"
            if not json_file.exists():
                raise FileNotFoundError(f"JSON file not found for {file.name}")
            shutil.copy2(json_file, json_dir / json_file.name)
            
    # Return split information
    split_info = {
        'splits': splits,
        'stats': {
            'total': total_files,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size
        }
    }
    
    return split_info

def setup_logging(log_dir: str) -> logging.Logger:
    """
    Setup logging configuration.
    
    Creates a logger with both file and console handlers.
    
    Args:
        log_dir: Directory to save log files
        
    Returns:
        Configured logger instance
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger('augmentation')
    logger.setLevel(logging.INFO)
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fh = logging.FileHandler(log_dir / f'augmentation_{timestamp}.log')
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def save_config(args: argparse.Namespace, output_dir: str):
    """
    Save configuration to JSON file.
    
    Useful for documenting the exact parameters used for a run.
    
    Args:
        args: Command line arguments from argparse
        output_dir: Directory to save the configuration file
    """
    config = vars(args)
    config_path = Path(output_dir) / 'augmentation_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def main():
    """
    Main function with improved workflow.
    
    Parses command line arguments and executes the requested operation:
    - Split dataset into train/val/test
    - Run augmentation pipeline
    - Debug augmentations on a single image
    """
    parser = argparse.ArgumentParser(description='Dataset preparation and augmentation pipeline')
    
    # Command selection
    parser.add_argument('-s','--split-data', action='store_true',
                      help='Split dataset into train/val/test')
    parser.add_argument('-a','--augment', action='store_true',
                      help='Run augmentation pipeline')
    parser.add_argument('-d', '--debug-image',
                      help='Debug augmentations for specific image')
    
    args = parser.parse_args()
    config.validate_config()
    
    # Setup logging
    logger = setup_logging(Path(config.DATA_CONFIG['output_dir']) / 'logs')
    
    try:
        if args.split_data:
            logger.info("Splitting dataset...")
            source_dir = Path(config.DATA_CONFIG['raw_images'])
            split_info = split_dataset(source_dir, config.DATA_CONFIG['split'])
            
            # Log the split counts using the stats dictionary
            logger.info(
                f"Dataset split complete: "
                f"train={split_info['stats']['train_size']}, "
                f"val={split_info['stats']['val_size']}, "
                f"test={split_info['stats']['test_size']}"
            )
            
        elif args.augment:
            logger.info("Starting augmentation pipeline...")
            
            # Train set augmentation
            train_augmenter = InstrumentDatasetAugmenter(
                image_dir=str(config.get_processed_data_path('train', 'images')),
                json_dir=str(config.get_processed_data_path('train', 'jsons')),
                output_image_dir=str(Path(config.AUGMENTATION_CONFIG['output_dir']) / 'train' / 'images'),
                output_json_dir=str(Path(config.AUGMENTATION_CONFIG['output_dir']) / 'train' / 'jsons'),
                max_size=config.AUGMENTATION_CONFIG['max_size']
            )
            train_augmenter.augment_dataset()
            
            # Validation set augmentation (if enabled)
            if config.DATA_CONFIG['split']['augment_validation']:
                val_augmenter = InstrumentDatasetAugmenter(
                    image_dir=str(Path(config.DATA_CONFIG['image_dir']) / 'val' / 'images'),
                    json_dir=str(Path(config.DATA_CONFIG['image_dir']) / 'val' / 'jsons'),
                    output_image_dir=str(config.get_augmented_data_path('train', 'images')),
                    output_json_dir=str(config.get_augmented_data_path('train', 'jsons')),
                    max_size=config.AUGMENTATION_CONFIG['max_size']
                )
                val_augmenter.augment_dataset()
            
            logger.info("Augmentation pipeline completed")
            
        elif args.debug_image:
            logger.info(f"Running debug mode for image: {args.debug_image}")
            debug_augmenter = InstrumentDatasetAugmenter(
                image_dir=os.path.dirname(args.debug_image),
                json_dir=config.DATA_CONFIG['raw_jsons'],
                output_image_dir=str(Path(config.AUGMENTATION_CONFIG['output_structure']['debug']['images'])),
                output_json_dir=str(Path(config.AUGMENTATION_CONFIG['output_structure']['debug']['jsons'])),
                max_size=config.AUGMENTATION_CONFIG['max_size']
            )
            debug_augmenter.debug_single_image(os.path.basename(args.debug_image))
            
        else:
            parser.print_help()
            
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()