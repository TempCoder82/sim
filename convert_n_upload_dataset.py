#!/usr/bin/env python3
"""
Florence-2 Dataset Converter

This script converts the augmented surgical instrument dataset to the format required by
Florence-2 model for training. It handles both object detection and segmentation tasks,
normalizing coordinates and formatting data according to model requirements.

Main features:
- Converts images and annotations to Hugging Face dataset format
- Normalizes bounding boxes and polygons to 0-999 coordinate space
- Handles pushing to Hugging Face Hub with sharding for large datasets
- Preserves task-specific formatting (OD and segmentation)
"""

import os
import json
from PIL import Image
from shapely.geometry import LineString
import datasets
from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import config
from huggingface_hub import login

class Florence2HFConverter:
    """
    Converts augmented dataset to Hugging Face format compatible with Florence-2 model.
    
    Handles coordinate normalization, format conversion, and dataset upload.
    """
    
    def __init__(self):
        """
        Initialize converter using configuration.
        
        Sets up paths and logging based on config settings.
        """
        self.cfg = config.DATA_CONFIG
        augment_cfg = config.AUGMENTATION_CONFIG
        
        # Use the correct paths from augmentation config
        base_aug_dir = Path(augment_cfg['output_dir'])
        self.train_image_dir = base_aug_dir / 'train' / 'images'
        self.train_json_dir = base_aug_dir / 'train' / 'jsons'
        self.val_image_dir = base_aug_dir / 'val' / 'images'
        self.val_json_dir = base_aug_dir / 'val' / 'jsons'
        
        # Setup logging
        self.logger = self.setup_logging()
        self._verify_directories()

    def _verify_directories(self):
        """
        Verify directories exist and count files.
        
        Logs the number of files found in each directory to help diagnose issues.
        """
        directories = {
            'Train Images': self.train_image_dir,
            'Train JSONs': self.train_json_dir,
            'Val Images': self.val_image_dir,
            'Val JSONs': self.val_json_dir
        }
        
        for name, dir_path in directories.items():
            if not dir_path.exists():
                self.logger.error(f"{name} directory not found: {dir_path}")
            else:
                file_count = len(list(dir_path.glob("*")))
                self.logger.info(f"{name}: Found {file_count} files in {dir_path}")

    def setup_logging(self) -> logging.Logger:
        """
        Setup logging configuration.
        
        Creates timestamped log files and configures console output.
        
        Returns:
            Logger instance configured for both file and console output
        """
        log_dir = Path(self.cfg['output_dir']) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger('dataset_converter')
        logger.setLevel(logging.INFO)
        
        # File handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fh = logging.FileHandler(log_dir / f'conversion_{timestamp}.log')
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

    def find_image_file(self, json_file: Path, image_dir: Path) -> Path:
        """
        Find corresponding image file trying all common formats.
        
        Args:
            json_file: Path to JSON annotation file
            image_dir: Directory containing image files
            
        Returns:
            Path to image file if found, None otherwise
        """
        base_name = json_file.stem
        
        # Try all common image formats
        for ext in ['.png', '.jpg', '.jpeg']:
            image_path = image_dir / f"{base_name}{ext}"
            if image_path.exists():
                return image_path
            
        return None

    def normalize_bbox(self, bbox: list, img_width: int, img_height: int) -> list:
        """
        Normalize bbox coordinates to 0-999 range.
        
        Args:
            bbox: Bounding box in [x, y, w, h] format
            img_width: Width of the image
            img_height: Height of the image
            
        Returns:
            Normalized bounding box in [x1, y1, x2, y2] format with coordinates in 0-999 range
        """
        x, y, w, h = bbox
        x1 = int((x / img_width) * 999)
        y1 = int((y / img_height) * 999)
        x2 = int(((x + w) / img_width) * 999)
        y2 = int(((y + h) / img_height) * 999)
        return [x1, y1, x2, y2]

    def normalize_polygon(self, polygon: list, img_width: int, img_height: int, target_points=40) -> list:
        """
        Normalize and simplify polygon coordinates to a 0-999 range.

        Args:
            polygon: List of (x, y) polygon coordinates
            img_width: Width of the image
            img_height: Height of the image
            target_points: Desired number of points after simplification
            
        Returns:
            Flattened list of normalized (x, y) coordinates in 0-999 range
        """
        if len(polygon) <= target_points:
            simplified_polygon = polygon  # No need to simplify if already under limit
        else:
            # Convert polygon to LineString for simplification
            polygon_line = LineString(polygon)

            # Adaptive tolerance search for target number of points
            low, high = 0, 20  # Adjust search range
            best_simplified = polygon

            while low <= high:
                mid = (low + high) / 2
                temp_simplified = polygon_line.simplify(mid, preserve_topology=True)
                temp_coords = list(temp_simplified.coords)

                if len(temp_coords) == target_points:
                    simplified_polygon = temp_coords
                    break
                elif len(temp_coords) > target_points:
                    low = mid + 0.1  # Increase tolerance to remove more points
                else:
                    high = mid - 0.1  # Decrease tolerance to keep more points

                if abs(len(temp_coords) - target_points) < abs(len(best_simplified) - target_points):
                    best_simplified = temp_coords  # Store best match found so far

            simplified_polygon = best_simplified  # Use the best-found simplification

        # Normalize to 0-999 range
        normalized = []
        for x, y in simplified_polygon:
            norm_x = int((x / img_width) * 999)
            norm_y = int((y / img_height) * 999)
            normalized.extend([norm_x, norm_y])

        return normalized

    def prepare_examples(self, split: str = "train", files=None):
        """
        Prepare examples for the dataset.
        
        Creates object detection and segmentation examples in Florence-2 format.
        
        Args:
            split: Dataset split ("train" or "val")
            files: Optional list of specific files to process
            
        Returns:
            Dictionary of examples with keys for each required field
        """
        examples = {
            "image": [],
            "label": [],
            "text_input": [],
            "task_type": [],
            "prompt": []
        }
        
        # Use the correct directory based on split
        if split == "train":
            image_dir = self.train_image_dir
            json_files = files if files is not None else sorted(list(self.train_json_dir.glob("*.json")))
        else:
            image_dir = self.val_image_dir
            json_files = sorted(list(self.val_json_dir.glob("*.json")))
        
        self.logger.info(f"Processing {len(json_files)} files for {split} split")
        skipped_files = 0
        
        for json_file in tqdm(json_files):
            try:
                # Find corresponding image file
                image_file = self.find_image_file(json_file, image_dir)
                
                if not image_file:
                    self.logger.warning(f"Image not found for {json_file}")
                    skipped_files += 1
                    continue
                
                # Get image dimensions without loading into memory
                with Image.open(image_file) as img:
                    img_width, img_height = img.size
                
                with open(json_file, "r") as f:
                    data = json.load(f)
                
                for instrument in data.get("instruments", []):
                    # Create OD example
                    od_suffix = f"{instrument['name']}"
                    bbox = self.normalize_bbox(instrument["bbox"], img_width, img_height)
                    for coord in bbox:
                        od_suffix += f"<loc_{coord}>"
                    
                    examples["image"].append(str(image_file))
                    examples["label"].append(od_suffix)
                    examples["text_input"].append("")
                    examples["task_type"].append("od")
                    examples["prompt"].append("<OD>")
                    
                    # Create REF_SEG example
                    ref_seg_suffix = "<seg>"  # Start with <seg>
                    poly_coords = self.normalize_polygon(
                        instrument["polygon"], 
                        img_width, 
                        img_height,
                        target_points=40  # Set the desired number of points
                    )
                    for i in range(0, len(poly_coords), 2):
                        x, y = poly_coords[i], poly_coords[i + 1]
                        ref_seg_suffix += f"<loc_{x}><loc_{y}>"
                    ref_seg_suffix += "</seg>"  # End with </seg>
                    
                    examples["image"].append(str(image_file))
                    examples["label"].append(ref_seg_suffix)
                    examples["text_input"].append(instrument["name"])
                    examples["task_type"].append("ref_seg")
                    examples["prompt"].append("<REFERRING_EXPRESSION_SEGMENTATION>")
                    
            except Exception as e:
                self.logger.error(f"Error processing {json_file}: {str(e)}")
                skipped_files += 1
                continue
        
        self.logger.info(f"Created {len(examples['image'])} examples")
        self.logger.info(f"Skipped {skipped_files} files")
        return examples

    def prepare_and_upload_dataset(self):
        """
        Prepare and upload dataset in shards.
        
        Processes validation set completely and training set in shards to handle large datasets.
        """
        self.logger.info("Starting dataset conversion...")
        
        try:
            # Process validation set first (no sharding needed)
            val_examples = self.prepare_examples("val")
            
            # Define features correctly using Features class
            features = Features({
                "image": HFImage(),
                "label": Value("string"),
                "text_input": Value("string"),
                "task_type": Value("string"),
                "prompt": Value("string")
            })
            
            # Create validation dataset
            val_dataset = Dataset.from_dict(
                val_examples,
                features=features
            )
            
            # Get all training JSON files
            train_json_files = sorted(list(self.train_json_dir.glob("*.json")))
            self.logger.info(f"Found {len(train_json_files)} training files")
            
            # Process training data in shards
            shard_size = 400  # Number of files per shard
            num_shards = (len(train_json_files) + shard_size - 1) // shard_size
            
            # Initialize first shard
            self.logger.info("Creating first training shard...")
            first_shard = train_json_files[:shard_size]
            first_examples = self.prepare_examples("train", first_shard)
            train_dataset = Dataset.from_dict(
                first_examples,
                features=features
            )
            
            # Create initial dataset dictionary
            dataset_dict = DatasetDict({
                "train": train_dataset,
                "val": val_dataset
            })
            
            # Push initial version to hub
            self.logger.info("Pushing initial version to hub...")
            dataset_dict.push_to_hub(
                self.cfg['dataset_name'], 
                private=True,
                token=self.cfg['hf_token']
            )
            
            # Process remaining shards
            for shard_idx in range(1, num_shards):
                start_idx = shard_idx * shard_size
                end_idx = min(start_idx + shard_size, len(train_json_files))
                shard_files = train_json_files[start_idx:end_idx]
                
                self.logger.info(f"Processing shard {shard_idx + 1}/{num_shards}")
                shard_examples = self.prepare_examples("train", shard_files)
                shard_dataset = Dataset.from_dict(
                    shard_examples,
                    features=features
                )
                
                # Append to existing dataset on hub
                self.logger.info(f"Pushing shard {shard_idx + 1} to hub...")
                shard_dataset.push_to_hub(
                    self.cfg['dataset_name'],
                    private=True,
                    token=self.cfg['hf_token'],
                    split=f"train_shard_{shard_idx}",
                )
                
                # Clear memory
                del shard_examples
                del shard_dataset
                
            self.logger.info("Dataset upload completed successfully")
                
        except Exception as e:
            self.logger.error(f"Error during dataset conversion: {str(e)}")
            raise

def main():
    """
    Main conversion function.
    
    Handles Hugging Face login and dataset conversion.
    """
    # Login to HuggingFace if token is provided
    if config.DATA_CONFIG.get('hf_token'):
        login(config.DATA_CONFIG['hf_token'])
    
    # Create converter
    converter = Florence2HFConverter()
    
    try:
        # Convert and upload dataset with sharding
        converter.prepare_and_upload_dataset()
        
    except Exception as e:
        converter.logger.error(f"Error in conversion pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
