"""
Model Evaluation Script
Checking how accurate the model really is.
Can run on the Food-101 dataset or my own test images.
"""

import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import os
from pathlib import Path
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm
import json

def load_model_and_processor(model_path="./my_final_dinov2_food101_model_FULL"):
    """Load up the model and processor"""
    print("Loading model and processor...")
    model = AutoModelForImageClassification.from_pretrained(model_path)
    processor = AutoImageProcessor.from_pretrained(model_path)
    model.eval()
    print("Model loaded successfully!")
    return model, processor

def predict_image(model, processor, image_path):
    """Run prediction on a single image"""
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
        
        return predicted_class_idx
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def evaluate_with_food101(model, processor, num_samples=None):
    """
    Test using the standard Food-101 dataset from HuggingFace.
    Need to have 'datasets' installed for this.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not installed. Install with: pip install datasets")
        return None
    
    print("Loading Food-101 test dataset from HuggingFace...")
    dataset = load_dataset("food101", split="test")
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} test images...")
    
    # Get label mappings
    id2label = model.config.id2label
    label2id = model.config.label2id
    
    true_labels = []
    predicted_labels = []
    
    for item in tqdm(dataset, desc="Processing images"):
        true_label = item['label']  # the actual correct label index
        image = item['image']
        
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Predict
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
        
        true_labels.append(true_label)
        predicted_labels.append(predicted_class_idx)
    
    return true_labels, predicted_labels, id2label

def evaluate_custom_dataset(model, processor, test_dir):
    """
    Run evaluation on my own custom dataset folder.
    Folder structure should be: test_dir/class_name/image.jpg
    """
    print(f"Loading custom test dataset from {test_dir}...")
    
    id2label = model.config.id2label
    label2id = model.config.label2id
    
    true_labels = []
    predicted_labels = []
    image_paths = []
    
    # Find all images
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"Error: Test directory {test_dir} does not exist!")
        return None
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    for ext in image_extensions:
        image_paths.extend(test_path.rglob(f'*{ext}'))
        image_paths.extend(test_path.rglob(f'*{ext.upper()}'))
    
    if not image_paths:
        print(f"No images found in {test_dir}")
        return None
    
    print(f"Found {len(image_paths)} images. Processing...")
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        # Get true label from folder name
        class_folder = img_path.parent.name
        if class_folder not in label2id:
            print(f"Warning: Class '{class_folder}' not in model labels. Skipping {img_path}")
            continue
        
        true_label_idx = label2id[class_folder]
        
        # Predict
        predicted_idx = predict_image(model, processor, img_path)
        if predicted_idx is not None:
            true_labels.append(true_label_idx)
            predicted_labels.append(predicted_idx)
    
    if not true_labels:
        print("No valid predictions made!")
        return None
    
    return true_labels, predicted_labels, id2label

def calculate_metrics(true_labels, predicted_labels, id2label):
    """Calculate and display metrics"""
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # config.id2label keys come back as ints from from_pretrained; normalize so lookups are safe
    id2label = {int(k): v for k, v in id2label.items()}

    # Overall accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY: {accuracy*100:.2f}%")
    print(f"{'='*60}\n")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, average=None, zero_division=0
    )
    
    # Overall metrics (macro and weighted averages)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='weighted', zero_division=0
    )
    
    print(f"Macro Average - Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")
    print(f"Weighted Average - Precision: {precision_weighted:.4f}, Recall: {recall_weighted:.4f}, F1: {f1_weighted:.4f}\n")
    
    # Show stats for each class
    print("Per-Class Metrics:")
    print(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    
    class_results = []
    for idx in range(len(id2label)):
        label_name = id2label[idx]
        class_results.append({
            'class': label_name,
            'precision': precision[idx],
            'recall': recall[idx],
            'f1': f1[idx],
            'support': support[idx]
        })
        print(f"{label_name:<30} {precision[idx]:<12.4f} {recall[idx]:<12.4f} {f1[idx]:<12.4f} {support[idx]:<10}")
    
    # Classification report
    print(f"\n{'='*60}")
    print("DETAILED CLASSIFICATION REPORT")
    print(f"{'='*60}\n")
    target_names = [id2label[i] for i in range(len(id2label))]
    print(classification_report(true_labels, predicted_labels, target_names=target_names, zero_division=0))
    
    # Confusion matrix (only show if it's small enough to read)
    if len(id2label) <= 20 or len(true_labels) < 1000:
        print(f"\n{'='*60}")
        print("CONFUSION MATRIX")
        print(f"{'='*60}\n")
        cm = confusion_matrix(true_labels, predicted_labels)
        print(cm)
    else:
        print(f"\nNote: Confusion matrix not displayed (too large: {len(id2label)} classes)")
        print("Top misclassifications can be analyzed from per-class metrics above.")
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'per_class': class_results
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate food classification model')
    parser.add_argument('--model_path', type=str, default='./my_final_dinov2_food101_model_FULL',
                        help='Path to model directory')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='Path to custom test dataset (folder structure: class_name/image.jpg)')
    parser.add_argument('--food101', action='store_true',
                        help='Use Food-101 test dataset from HuggingFace')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (for Food-101, None = all)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Load model
    model, processor = load_model_and_processor(args.model_path)
    
    # Evaluate
    if args.food101:
        results = evaluate_with_food101(model, processor, args.num_samples)
    elif args.test_dir:
        results = evaluate_custom_dataset(model, processor, args.test_dir)
    else:
        print("Error: Please specify either --food101 or --test_dir")
        print("Example: python evaluate_model.py --food101 --num_samples 1000")
        print("Example: python evaluate_model.py --test_dir ./test_data")
        return
    
    if results is None:
        print("Evaluation failed!")
        return
    
    true_labels, predicted_labels, id2label = results
    
    if len(true_labels) == 0:
        print("No valid predictions to evaluate!")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(true_labels, predicted_labels, id2label)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == '__main__':
    main()

