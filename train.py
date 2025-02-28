#!/usr/bin/env python3
import os
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor, AdamW, get_scheduler
from datasets import load_dataset, concatenate_datasets
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple

def load_sharded_dataset(dataset_name: str) -> Dict[str, Dataset]:
    """
    Load and combine sharded datasets.
    For training, combine the "train" split and any "train_shard_*" splits.
    For validation, only load the "val" split since no shards are provided.
    """
    # Load all splits from the hub
    all_splits = load_dataset(dataset_name)
    
    # Handle training shards
    train_splits = []
    if "train" in all_splits:
        train_splits.append(all_splits["train"])
    for split_name in all_splits.keys():
        if split_name.startswith("train_shard_"):
            train_splits.append(all_splits[split_name])
            
    if len(train_splits) == 0:
        raise ValueError("No training splits found!")
    train_full = concatenate_datasets(train_splits)
    logging.info("Training dataset loaded with %d examples.", len(train_full))
    
    # For validation, only load the "val" split as no validation shards exist.
    if "val" not in all_splits:
        raise ValueError("No validation splits found!")
    val_full = all_splits["val"]
    logging.info("Validation dataset loaded with %d examples.", len(val_full))
    
    return {
        "train": train_full,
        "val": val_full
    }


class MultiTaskDataset(Dataset):
    """Dataset class for Florence-2 multi-task training"""
    def __init__(self, data, split='train'):
        self.data = data
        self.split = split
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Get task-specific prompt
        prompt = example['prompt']
        if example['task_type'] == 'ref_seg' and example['text_input']:
            prompt += example['text_input']
            
        # Get image
        image = example['image']
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        return prompt, example['label'], image, example['task_type']
    
def collate_fn(batch, processor):
    """
    Generic collate function for both tasks
    
    Returns:
        tuple: (inputs, targets, task_types) with image sizes included for post-processing
    """
    prompts_or_questions, targets, images, task_types = zip(*batch)
    
    # Get image sizes for later use in post-processing
    image_sizes = [(img.width, img.height) for img in images]
    
    inputs = processor(
        text=list(prompts_or_questions),
        images=list(images),
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding='max_length'
    )
    
    # Store image sizes and task types for later post-processing
    inputs["image_sizes"] = image_sizes
    inputs["task_types"] = task_types
        
    return inputs, targets

class TrainingMonitor:
    """Monitor and visualize training progress"""
    def __init__(self, save_dir: str = "training_metrics"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.lr_history = []
        self.best_val_loss = float('inf')
        
    def update(self, train_loss: float, val_loss: float, current_lr: float):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.lr_history.append(current_lr)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            return True
        return False
        
    def plot_metrics(self, epoch: int):
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Loss vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Learning rate plot
        plt.subplot(1, 2, 2)
        plt.plot(self.lr_history)
        plt.title('Learning Rate vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'training_metrics_epoch_{epoch}.png')
        plt.close()

def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fh = logging.FileHandler(log_dir / f'training_{timestamp}.log')
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def evaluate_model(model, processor, val_loader, device, logger):
    """Simply evaluate loss on validation set"""
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating"):
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                
            # Tokenize targets
            labels = processor.tokenizer(
                text=list(targets),
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False
            ).input_ids.to(device)
            
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels
            )
            
            val_loss += outputs.loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    logger.info(f"Validation Loss: {avg_val_loss:.4f}")
    
    return avg_val_loss

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    config: Dict
) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    """Train the model"""
    
    # Setup
    device = config['device']
    model = model.to(device)
    save_dir = Path(config['output_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize training components
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    num_training_steps = config['epochs'] * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    # Initialize monitor and logger
    monitor = TrainingMonitor(save_dir / 'metrics')
    logger = setup_logging(save_dir / 'logs')
    
    logger.info("Starting training")
    logger.info(f"Training on device: {device}")
    logger.info(f"Number of training examples: {len(train_loader.dataset)}")
    logger.info(f"Number of validation examples: {len(val_loader.dataset)}")
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        
        # Training loop
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{config['epochs']}")
        for batch in progress_bar:
            inputs, targets = batch
            
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                
            # Tokenize targets
            labels = processor.tokenizer(
                text=list(targets),
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False
            ).input_ids.to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        avg_val_loss = evaluate_model(model, processor, val_loader, device, logger)
        
        # Update training monitor
        is_best = monitor.update(
            avg_train_loss,
            avg_val_loss,
            optimizer.param_groups[0]['lr']
        )
        monitor.plot_metrics(epoch + 1)
        
        logger.info(f"Epoch {epoch + 1}/{config['epochs']}")
        logger.info(f"Average Training Loss: {avg_train_loss:.4f}")
        logger.info(f"Average Validation Loss: {avg_val_loss:.4f}")
        
        # Save checkpoints
        if is_best:
            logger.info(f"New best validation loss: {avg_val_loss:.4f}")
            model.save_pretrained(save_dir / 'best_model')
            processor.save_pretrained(save_dir / 'best_model')
        
        if (epoch + 1) % config['save_frequency'] == 0:
            output_dir = save_dir / f'checkpoint_epoch_{epoch+1}'
            output_dir.mkdir(exist_ok=True)
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
    
    return model, processor

def main():
    # Import config here to avoid circular imports
    import config
    
    # Load dataset
    dataset = load_sharded_dataset(config.DATA_CONFIG['dataset_name'])
    
    # Create datasets
    train_dataset = MultiTaskDataset(dataset['train'])
    val_dataset = MultiTaskDataset(dataset['val'])
    
    # Initialize model and processor
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_CONFIG['name'],
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(
        config.MODEL_CONFIG['name'],
        trust_remote_code=True
    )
    
    # Create dataloaders with proper collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAINING_CONFIG['batch_size'],
        collate_fn=lambda batch: collate_fn(batch, processor),
        num_workers=config.TRAINING_CONFIG['num_workers'],
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.TRAINING_CONFIG['batch_size'],
        collate_fn=lambda batch: collate_fn(batch, processor),
        num_workers=config.TRAINING_CONFIG['num_workers']
    )
    
    # Optional: Freeze vision tower
    if config.MODEL_CONFIG.get('freeze_vision', False):
        print("Freezing vision tower parameters")
        for param in model.vision_tower.parameters():
            param.requires_grad = False
    
    # Train model
    trained_model, trained_processor = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        processor=processor,
        config=config.TRAINING_CONFIG
    )
    
    # Save final model
    if config.TRAINING_CONFIG.get('push_to_hub', False):
        trained_model.push_to_hub(config.TRAINING_CONFIG['hub_model_id'])
        trained_processor.push_to_hub(config.TRAINING_CONFIG['hub_model_id'])

if __name__ == "__main__":
    main()
