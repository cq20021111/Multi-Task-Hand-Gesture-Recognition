import torch
import numpy as np
from model import MultiTaskHandNet
from dataloader import get_dataloader
from utils import calculate_iou_box, calculate_iou_mask, calculate_dice_mask, calculate_detection_accuracy_at_iou
import os
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Use Config class instead of command line arguments
class Config:
    # Get the directory where the current script is located
    _src_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory
    _base_dir = os.path.dirname(_src_dir)
    
    # Automatically derive data directory and model path
    data_dir = os.path.join(_base_dir, 'data')
    model_path = os.path.join(_base_dir, 'weights', 'best_model.pth')
    results_dir = os.path.join(_base_dir, 'results')
    batch_size = 32

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """
    Plot and save the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

def evaluate_dataset(loader, model, device, dataset_name="Test", save_cm=False, results_dir=None):
    """
    Helper function: evaluate performance on a specified dataset
    """
    total_acc = 0.0
    total_iou_box = 0.0
    total_det_acc_50 = 0.0
    total_iou_mask = 0.0
    total_dice_mask = 0.0
    num_batches = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, masks, bboxes, labels in loader:
            images = images.to(device)
            masks = masks.to(device)
            bboxes = bboxes.to(device)
            labels = labels.to(device)
            
            # Forward pass
            cls_logits, bbox_preds, mask_logits = model(images)
            
            # --- Calculate classification accuracy ---
            _, preds = torch.max(cls_logits, 1)
            acc = (preds == labels).float().mean().item()
            total_acc += acc
            
            # Collect predictions for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # --- Calculate bounding box IoU and Detection Accuracy ---
            bbox_preds_denorm = bbox_preds * 224.0
            total_iou_box += calculate_iou_box(bbox_preds_denorm, bboxes)
            total_det_acc_50 += calculate_detection_accuracy_at_iou(bbox_preds_denorm, bboxes, iou_threshold=0.5)
            
            # --- Calculate segmentation mask IoU and Dice ---
            total_iou_mask += calculate_iou_mask(mask_logits, masks)
            total_dice_mask += calculate_dice_mask(mask_logits, masks)
            
            num_batches += 1
            
    if num_batches > 0:
        # Calculate Macro-averaged F1 Score based on collected predictions and labels
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f"[{dataset_name}] Classification Overall top-1 accuracy: {total_acc/num_batches:.4f}")
        print(f"[{dataset_name}] Classification Macro F1 Score: {macro_f1:.4f}")
        print(f"[{dataset_name}] Box IoU: {total_iou_box/num_batches:.4f}")
        print(f"[{dataset_name}] Box Detection Accuracy @ IoU=0.5: {total_det_acc_50/num_batches:.4f}")
        print(f"[{dataset_name}] Mask IoU: {total_iou_mask/num_batches:.4f}")
        print(f"[{dataset_name}] Mask Dice: {total_dice_mask/num_batches:.4f}")
        
        if save_cm and results_dir:
            os.makedirs(results_dir, exist_ok=True)
            # Get class names (assuming loader.dataset is HandGestureDataset or Subset)
            dataset = loader.dataset
            if hasattr(dataset, 'dataset'): # Handle Subset
                classes = dataset.dataset.classes
            else:
                classes = dataset.classes
                
            cm_path = os.path.join(results_dir, f'confusion_matrix_{dataset_name.lower()}.png')
            plot_confusion_matrix(all_labels, all_preds, classes, cm_path)
            
    else:
        print(f"[{dataset_name}] No batches processed.")

def evaluate():
    """
    Model evaluation function.
    Loads the trained model and calculates accuracy and IoU metrics on train and test sets.
    """
    args = Config()
    
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- 1. Load data ---
    # Load test set
    test_loader = get_dataloader(args.data_dir, split='test', batch_size=args.batch_size, shuffle=False)
    # Load training and validation sets
    # Use the same val_split (0.13) as during training to ensure consistent split
    train_loader, val_loader = get_dataloader(args.data_dir, split='training', batch_size=args.batch_size, shuffle=False, val_split=0.13)
    
    print(f"Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # --- 2. Load model ---
    model = MultiTaskHandNet(num_classes=10).to(device)
    if not os.path.exists(args.model_path):
        print(f"Model path {args.model_path} does not exist.")
        return

    # Load weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval() # Set to evaluation mode
    
    # --- 3. Evaluation loop ---
    print("\n--- Evaluating on Training Set ---")
    evaluate_dataset(train_loader, model, device, dataset_name="Train", save_cm=False)
    
    print("\n--- Evaluating on Validation Set ---")
    evaluate_dataset(val_loader, model, device, dataset_name="Validation", save_cm=True, results_dir=args.results_dir)
    
    print("\n--- Evaluating on Test Set ---")
    evaluate_dataset(test_loader, model, device, dataset_name="Test", save_cm=True, results_dir=args.results_dir)

if __name__ == '__main__':
    evaluate()