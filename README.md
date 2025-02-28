# Surgical Instrument Detection & Segmentation with Florence-2

This repository contains a fine-tuning implementation for Microsoft's Florence-2 vision-language model (VLM) to perform simultaneous object detection and segmentation of surgical instruments. The implementation demonstrates how a single model can be fine-tuned to excel at identifying and localizing specific surgical tools in medical imagery.

## Project Overview

Florence-2 is a powerful vision-language model with remarkable multi-task capabilities. This project leverages these capabilities by fine-tuning the model on a dataset of surgical instruments to perform two key tasks:

1. **Object Detection (OD)**: Identifying and localizing five types of surgical instruments with bounding boxes
2. **Segmentation (SEG)**: Creating precise polygon segmentation masks for each instrument

The implementation uses a multi-task learning approach, allowing the model to learn both tasks simultaneously through a unified training pipeline.

## Surgical Instruments

The model is trained to recognize five specific surgical instruments:

- NEEDLE HOLDER MAYO HEGAR 7"
- LANE DISSECTING FORCEP 2 IN 3 TOOTHED 5.5"
- LAHEY FORCEP
- MAYO SCISSOR STRAIGHT GOLD HANDLED 5.5"
- FORCEP LITTLEWOOD

## Repository Structure

```
├── config.py                   # Configuration settings for paths, model, and training
├── split_n_augment_dataset.py  # Dataset splitting and augmentation functionality
├── convert_n_upload_dataset.py # Converting to HF dataset format and uploading
├── train.py                    # Model training implementation
├── test_sim_model.ipynb        # Jupyter notebook for testing the model
└── test/                       # Sample inference results and test images
    ├── images/                 # Test images
    └── results/                # Visualization outputs and JSON results
        ├── output1/            # First set of output results
        └── output2/            # Second set of output results
```

## Technical Approach

### Data Preparation Pipeline

1. **Dataset Splitting** (`split_n_augment_dataset.py`):
   - Divides the raw dataset into train/validation/test splits 
   - Uses configurable ratios (default: 80/15/5%)
   - Ensures paired image and JSON annotation files are properly handled

2. **Data Augmentation** (`split_n_augment_dataset.py`):
   - Implements 10 different augmentation techniques using Albumentations
   - Preserves both bounding boxes and segmentation polygons during transformations
   - Includes brightness/contrast adjustments, rotation, flips, HSV shifts, blur, noise, and more
   - Resizes images to a maximum dimension of 1024px while maintaining aspect ratio

3. **Dataset Conversion** (`convert_n_upload_dataset.py`):
   - Converts the augmented dataset to Hugging Face dataset format
   - Normalizes coordinates to 0-999 range as required by Florence-2
   - Implements **Adaptive Polygon Simplification** to reduce polygon point count to 40 points
   - Formats data according to Florence-2's task-specific prompts
   - Uploads data in shards for handling large datasets

### Adaptive Polygon Simplification

A key technical innovation in this project is the implementation of adaptive polygon simplification to address Florence-2's maximum sequence length limitation:

- Original segmentation polygons contained 100+ points, exceeding the models's maximum token capacity of 1024 
- Implemented binary search approach to find optimal simplification tolerance
- Based on the Douglas-Peucker algorithm via Shapely
- Targets exactly 40 points per polygon while preserving shape accuracy
- Reduces sequence length while maintaining segmentation quality

### Model Training

The `train.py` script implements a multi-task learning approach:

- Uses Florence-2-base-ft as the foundation model
- Freezes vision tower to focus learning on cross-modal connections
- Implements custom collate function to handle both tasks
- Uses AdamW optimizer with linear learning rate schedule
- Implements checkpointing based on validation loss
- Visualizes training progress with loss curves

### Inference

The `test_sim_model.ipynb` notebook provides an interactive way to test the model:

- Allows you to load and test your fine-tuned model in a Jupyter environment
- Processes single images interactively
- Supports both object detection and segmentation tasks
- Maps generic detections to specific instrument labels
- Creates visualizations with bounding boxes and segmentation masks
- Saves raw model outputs and processed results in JSON format
- Generates combined visualizations showing both bounding boxes and segmentation masks

## Sample Results

The repository includes sample inference results in the `test/results` directory. These examples demonstrate the model's capabilities on the surgical instrument image(s):

- **Object Detection**: Each detected instrument is highlighted with a bounding box and labeled with its specific type (`*_od.png` files)
- **Segmentation**: Precise polygon masks around each instrument with color-coding (`*_seg.png` files)
- **Combined Visualization**: Bounding boxes and segmentation masks shown together (`*_combined.png` files)
- **Raw Results**: JSON files with detailed detection coordinates and model outputs (`*_results.json` files)

These sample results provide a concrete demonstration of the model's performance to help us understand the expected output format when running inference.

## Requirements

```
# Core dependencies
torch
torchvision
transformers
pillow
numpy

# Dataset and data handling
datasets
pyarrow
huggingface_hub

# Data augmentation
albumentations
opencv-python
opencv-python-headless  # For environments without GUI

# Model specific
flash_attn  # For efficient attention computation
einops      # For tensor operations
timm        # For vision models

# Visualization and logging
matplotlib
tqdm
jupyter
ipywidgets  # For interactive notebook elements
```

## Usage Instructions

### 1. Configure the Environment

Copy the `env-example.txt` to `.env` and add Hugging Face token:

```
cp env-example.txt .env
# Edit .env with your HF_TOKEN
```

Edit the `config.py` file to set paths and parameters:

```python
# Set your paths
BASE_DIR = Path("/path/to/your/project")
DATA_DIR = BASE_DIR / "content"
```

### 2. Split and Augment Dataset

```bash
python split_n_augment_dataset.py --split-data
python split_n_augment_dataset.py --augment
```

### 3. Convert and Upload Dataset

```bash
python convert_n_upload_dataset.py
```

### 4. Train the Model

```bash
python train.py
```

### 5. Run Inference Using the Notebook

Launch the Jupyter notebook server and open `test_sim_model.ipynb`:

```bash
jupyter notebook
```

In the notebook, you can:
- Load your trained model
- Choose between object detection, segmentation, or both
- Process single images 
- Visualize and save the results interactively
- Analyze raw model outputs

### 6. Review Sample Results

To understand what to expect from the model, examine the sample output in the `test/results` directory. These include various visualization types and raw detection data that match what you'll get when running inference.

## Limitations and Future Work

While this implementation demonstrates the potential of fine-tuning Florence-2 for surgical instrument detection and segmentation, there are several areas for improvement:

1. **Metrics Implementation**: 
   - Add proper evaluation metrics for object detection (mAP, IoU)
   - Implement segmentation metrics (Dice coefficient, pixel accuracy)

2. **Test Set Evaluation**:
   - Conduct thorough evaluation on held-out test data
   - Report precision/recall curves and confusion matrices

3. **Training Improvements**:
   - Extend training time with larger batch sizes
   - Implement learning rate finder and proper scheduler
   - Test unfreezing vision tower after initial training

4. **Data Handling and known issues**:
   - Due to a data saving error and oversight, the object detection performance is sub-optimal. The mistake is simple to correct: all bounding boxes per image should be saved in the training data. Unfortunately, while preparing both segmentation and object detection datasets together, only one object detection label per image was saved, which penalized the model even when it was able to detect multiple objects.
   - The object detection task is relatively simpler compared to segmentation as it doesn't hit the max sequence limit.
   - Segmentation dataset formation was particularly challenging as there is no documentation that explicitly shows how to properly format this data for Florence-2.
   - Despite these challenges, the small training outcome shows promising results with segmentation.
   - Tuning the token-reducing algorithm will be another area to explore; the current value of 40 points per polygon was chosen somewhat arbitrarily.


5. **Code Structure**:
   - Refactor into more modular components with better separation of concerns
   - Add comprehensive unit tests
   - Implement proper logging and experiment tracking

6. **Hyperparameter Optimization**:
   - Conduct systematic hyperparameter search
   - Test different prompt formulations
   - Experiment with different model sizes (base vs. large)

7. **Deployment**:
   - Add model quantization for deployment
   - Implement TensorRT or ONNX conversion for inference speed
   - Create a simple API wrapper

## Technical Insights

### Why Florence-2?

Florence-2 offers several advantages for this application:

1. **Multi-task Capability**: Single model for both detection and segmentation
2. **Zero-shot Generalization**: Better adaptation to variations in surgical instruments
3. **Lightweight Design**: More suitable for potential deployment in medical settings
4. **Prompt-based Interface**: Flexible control through simple text prompts

### Challenges Addressed

1. **Sequence Length Limitation**: The adaptive polygon simplification technique successfully addresses the 1024 token limit by reducing polygon coordinates while preserving shape accuracy.

2. **Surgical Instrument Specificity**: The model is guided to recognize specific surgical instruments through explicit naming and visualization during training.

## Conclusion

This project demonstrates the feasibility of fine-tuning Florence-2 for specialized medical imaging tasks. The multi-task approach allows a single model to perform both object detection and segmentation with reasonable accuracy. While there are limitations to the current implementation, the foundation provided here offers a solid starting point for further refinement and deployment in real-world medical settings.

## Acknowledgments

- Microsoft for developing the Florence-2 model
- Hugging Face for the transformers library and dataset infrastructure
- The Albumentations team for their excellent image augmentation library
