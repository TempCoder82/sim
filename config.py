"""
Configuration for Florence-2 pipeline including:
- Directory structure
- Data processing settings
- Model parameters
- Training configuration
"""

import torch
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path("/home/")
DATA_DIR = BASE_DIR / "content"
OUTPUT_DIR = DATA_DIR / "outputs"
AUG_DIR = OUTPUT_DIR / "augmented"

# Raw data location
RAW_DATA_DIR = DATA_DIR / "test"
RAW_IMAGES = RAW_DATA_DIR / "images"
RAW_JSONS = RAW_DATA_DIR / "jsons"

# Data configuration
DATA_CONFIG = {
    # Directory structure
    'raw_dir': str(RAW_DATA_DIR),
    'raw_images': str(RAW_IMAGES),
    'raw_jsons': str(RAW_JSONS),
    'processed_dir': str(DATA_DIR / "processed"),
    'output_dir': str(OUTPUT_DIR / "data"), 
    'aug_dir': str(AUG_DIR), 
    
    # Split settings
    'split': {
        'train_ratio': 0.8,
        'val_ratio': 0.15,
        'test_ratio': 0.05,
        'augment_validation': False,  # Whether to augment validation set
        'seed': 42  # For reproducibility
    },
    
    # HuggingFace settings
    'hf_token': os.getenv("HF_TOKEN", ""),  # Get token from .env file
    'dataset_name': "Source82/SIM",  # e.g., "your-username/dataset-name"
    
    # Dataset metadata
    'dataset_description': "Instrument dataset with object detection and segmentation annotations",
    'license': "MIT",
    'tags': ["computer-vision", "object-detection", "segmentation"],
    'version': "1.0.0"
}

# Augmentation configuration
AUGMENTATION_CONFIG = {
    # Output structure
    'output_dir': str(OUTPUT_DIR / "augmented"),
    'output_structure': {
        'train': {
            'images': 'train/images',
            'jsons': 'train/jsons'
        },
        'val': {
            'images': 'val/images',
            'jsons': 'val/jsons'
        },
        'debug': {
            'images': 'debug/images',
            'jsons': 'debug/jsons',
            'plots': 'debug/plots'
        }
    },
    
    # General settings
    'max_size': 1024,
    'save_original': True,
    
    # Augmentation parameters
    'augmentations': {
        'brightness': {
            'brightness_limit': 0.2,
            'contrast_limit': 0.2
        },
        'rotate': {
            'limit': 30
        },
        'hflip': {},
        'vflip': {},
        'hsv': {
            'hue_shift_limit': 20,
            'sat_shift_limit': 30,
            'val_shift_limit': 20
        },
        'blur': {
            'blur_limit': 5
        },
        'noise': {
            'std_range': (0.1, 0.2)
        },
        'sharpen': {
            'alpha': (0.2, 0.5),
            'lightness': (0.5, 1.0)
        },
        'rgbshift': {
            'r_shift_limit': 20,
            'g_shift_limit': 20,
            'b_shift_limit': 20
        },
        'clahe': {
            'clip_limit': 4.0,
            'tile_grid_size': (8, 8)
        }
    }
}

# Model configuration
MODEL_CONFIG = {
    'name': "microsoft/Florence-2-base-ft",
    'revision': 'refs/pr/6',
    'trust_remote_code': True,
    'freeze_vision': True,  # Whether to freeze vision tower
    'prompts': {
        'od': "<OD>",
        'ref_seg': "<REFERRING_EXPRESSION_SEGMENTATION>",
        'caption': "<MORE_DETAILED_CAPTION>"
    }
}

# Training configuration
TRAINING_CONFIG = {
    # Output structure
    'output_dir': str(OUTPUT_DIR / "training"),
    'output_structure': {
        'checkpoints': 'checkpoints',
        'best_model': 'best_model',
        'logs': 'logs',
        'metrics': 'metrics'
    },
    
    # Training parameters
    'batch_size': 6,
    'epochs': 10,
    'learning_rate': 1e-6,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'save_frequency': 1,  # Save every N epochs
    'eval_frequency': 1,  # Run evaluation every N epochs
    
    # Optimizer settings
    'optimizer': {
        'name': 'adamw',
        'weight_decay': 0.01,
        'beta1': 0.9,
        'beta2': 0.999
    },
    
    # Learning rate scheduler
    'scheduler': {
        'name': 'linear',
        'warmup_steps': 0,
        'num_training_steps': None  # Set during training
    },
    
    # HuggingFace integration
    'push_to_hub': True,
    'hub_model_id': "Source82/SIM-Model"
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'log_dir': str(OUTPUT_DIR / "logs")
}

def get_processed_data_path(split: str, data_type: str) -> Path:
    """Get path for processed data (after splitting)"""
    return Path(DATA_CONFIG['processed_dir']) / split / data_type

def get_augmented_data_path(split: str, data_type: str) -> Path:
    """Get path for augmented data"""
    return Path(AUGMENTATION_CONFIG['output_dir']) / split / data_type

def create_directories():
    """Create all necessary directories from configs"""
    directories = [
        # Raw data
        RAW_IMAGES,
        RAW_JSONS,
        
        # Processed data
        get_processed_data_path('train', 'images'),
        get_processed_data_path('train', 'jsons'),
        get_processed_data_path('val', 'images'),
        get_processed_data_path('val', 'jsons'),
        get_processed_data_path('test', 'images'),
        get_processed_data_path('test', 'jsons'),
        
        # Augmented data
        get_augmented_data_path('train', 'images'),
        get_augmented_data_path('train', 'jsons'),
        get_augmented_data_path('val', 'images'),
        get_augmented_data_path('val', 'jsons'),
        
        # Training outputs
        Path(TRAINING_CONFIG['output_dir']),
        
        # Logs
        Path(LOGGING_CONFIG['log_dir']),
        
        # Additional required directories
        OUTPUT_DIR,
        OUTPUT_DIR / "data",
        OUTPUT_DIR / "data" / "logs",
        Path(DATA_CONFIG['output_dir']),
        Path(DATA_CONFIG['output_dir']) / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def validate_config():
    """Validate configuration settings"""
    # Create necessary directories first
    create_directories()

    # Validate split ratios
    split_ratios = DATA_CONFIG['split']
    total_ratio = sum([split_ratios[k] for k in ['train_ratio', 'val_ratio', 'test_ratio']])
    assert abs(total_ratio - 1.0) < 1e-6, f"Split ratios must sum to 1.0, got {total_ratio}"
    
    # Validate paths
    assert RAW_IMAGES.exists(), f"Raw images directory not found: {RAW_IMAGES}"
    assert RAW_JSONS.exists(), f"Raw jsons directory not found: {RAW_JSONS}"
    
