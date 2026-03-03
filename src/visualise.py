import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from model import MultiTaskHandNet
from dataloader import get_dataloader
from utils import calculate_iou_box, calculate_iou_mask

# Use Config class instead of command line arguments
class Config:
    # Get the directory where the current script is located
    _src_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory
    _base_dir = os.path.dirname(_src_dir)
    
    data_dir = os.path.join(_base_dir, 'data')
    model_path = os.path.join(_base_dir, 'weights', 'best_model.pth')
    output_dir = os.path.join(_base_dir, 'results')
    batch_size = 4 # Visualize 4 samples

def calculate_single_metrics(pred_class, true_class, pred_prob, pred_bbox, true_bbox, pred_mask, true_mask):
    """
    Calculate comprehensive score for a single sample
    """
    # 1. Is category correct (0 or 1)
    cls_score = 1.0 if pred_class == true_class else 0.0
    
    # 2. Box IoU
    # Convert to batch format to reuse existing function, or calculate separately
    pred_bbox_tensor = torch.tensor(pred_bbox).unsqueeze(0)
    true_bbox_tensor = torch.tensor(true_bbox).unsqueeze(0)
    # calculate_iou_box expects unnormalized predicted boxes, since they are already denormalized before passing in, use directly
    box_iou = calculate_iou_box(pred_bbox_tensor, true_bbox_tensor)
    
    # 3. Mask IoU
    # Convert to batch format, dimension (1, 1, H, W)
    pred_mask_tensor = torch.tensor(pred_mask).unsqueeze(0).unsqueeze(0)
    true_mask_tensor = torch.tensor(true_mask).unsqueeze(0).unsqueeze(0)
    
    # Note: calculate_iou_mask internally has sigmoid and > 0.5 logic,
    # but we are passing in already processed binary masks (0/1).
    # To reuse, we need to bypass its internal sigmoid, or write a simple single image IoU calculation ourselves
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    mask_iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Comprehensive score (you can adjust weights as needed)
    # Here we simply combine classification correctness, Box IoU, Mask IoU and confidence
    # If classification is wrong, score drops significantly
    total_score = (cls_score * 0.33) + (box_iou * 0.33) + (mask_iou * 0.33) * pred_prob
    
    return total_score, cls_score, box_iou, mask_iou

def visualize_predictions():
    """
    Visualize model prediction results.
    Draw a batch from the test set, displaying:
    1. Original Image
    2. Ground Truth Mask
    3. Predicted Mask
    4. Ground Truth BBox (Green Box) vs Predicted BBox (Red Box)
    5. True Class vs Predicted Class
    """
    args = Config()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load data
    # In order to find the best and worst, we need to load the entire test set, or at least a large batch of data
    test_loader = get_dataloader(args.data_dir, split='test', batch_size=64, shuffle=True)
    images, masks, bboxes, labels = next(iter(test_loader))
    
    images = images.to(device)
    
    # 2. Load model
    model = MultiTaskHandNet(num_classes=10).to(device)
    if not os.path.exists(args.model_path):
        print(f"Model path {args.model_path} does not exist.")
        return
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # 3. Get prediction results
    with torch.no_grad():
        cls_logits, bbox_preds, mask_logits = model(images)
        
    # Process prediction results
    # Calculate probabilities
    probs = torch.nn.functional.softmax(cls_logits, dim=1)
    # Get maximum probability and corresponding index
    max_probs, cls_preds = torch.max(probs, 1)
    
    mask_preds = (torch.sigmoid(mask_logits) > 0.5).float()
    
    # Denormalize images for display
    # ImageNet mean/std
    # Extract the first 3 channels (RGB) for visualization
    rgb_images = images[:, :3, :, :]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    images_denorm = rgb_images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    # Class names
    class_names = [
        'call', 'dislike', 'like', 'ok', 'one',
        'palm', 'peace', 'rock', 'stop', 'three'
    ]
    
    # Evaluate all samples and calculate scores
    sample_scores = []
    num_samples_to_evaluate = images.shape[0]
    
    for i in range(num_samples_to_evaluate):
        bbox_true = bboxes[i].cpu().numpy()
        bbox_pred = bbox_preds[i].cpu().numpy() * 224.0
        mask_true = masks[i, 0].cpu().numpy()
        mask_pred_bin = mask_preds[i, 0].cpu().numpy()
        label_true = labels[i].item()
        label_pred = cls_preds[i].item()
        prob_pred = max_probs[i].item()
        
        score, cls_s, box_iou, mask_iou = calculate_single_metrics(
            label_pred, label_true, prob_pred, bbox_pred, bbox_true, mask_pred_bin, mask_true
        )
        
        sample_scores.append({
            'index': i,
            'score': score,
            'img': images_denorm[i].cpu().permute(1, 2, 0).numpy(),
            'mask_true': mask_true,
            'mask_pred': mask_pred_bin,
            'bbox_true': bbox_true,
            'bbox_pred': bbox_pred,
            'label_true': class_names[label_true],
            'label_pred': class_names[label_pred],
            'prob': prob_pred,
            'metrics': f"Cls:{int(cls_s)} BoxIoU:{box_iou:.2f} MaskIoU:{mask_iou:.2f} Score:{score:.2f}"
        })
        
    # Sort by score
    sample_scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Select the best 2 and worst 2
    selected_samples = []
    if len(sample_scores) >= 4:
        selected_samples.extend(sample_scores[:2]) # Top 2
        selected_samples.extend(sample_scores[-2:]) # Bottom 2
    else:
        selected_samples = sample_scores # Fallback if batch is too small

    # 4. Plot
    num_display = len(selected_samples)
    # Adjust canvas size
    fig, axes = plt.subplots(num_display, 3, figsize=(15, 4 * num_display))
    
    for row_idx, sample_data in enumerate(selected_samples):
        img = sample_data['img']
        mask_true = sample_data['mask_true']
        mask_pred = sample_data['mask_pred']
        bbox_true = sample_data['bbox_true']
        bbox_pred = sample_data['bbox_pred']
        label_true = sample_data['label_true']
        label_pred = sample_data['label_pred']
        prob_pred = sample_data['prob']
        metrics_str = sample_data['metrics']
        
        # Determine if it's best or worst
        if row_idx < 2:
            rank_str = f"BEST"
        else:
            rank_str = f"WORST"
        
        # --- Column 1: Original Image with BBoxes ---
        ax = axes[row_idx, 0] if num_display > 1 else axes[0]
        ax.imshow(img)
        title = f"{rank_str}\nTrue: {label_true} | Pred: {label_pred} ({prob_pred:.2%})\n{metrics_str}"
        ax.set_title(title, fontsize=10)
        
        # Draw True Box (Green)
        rect_true = plt.Rectangle((bbox_true[0], bbox_true[1]), 
                                  bbox_true[2] - bbox_true[0], 
                                  bbox_true[3] - bbox_true[1], 
                                  fill=False, edgecolor='green', linewidth=2, label='True Box')
        ax.add_patch(rect_true)
        
        # Draw Pred Box (Red)
        rect_pred = plt.Rectangle((bbox_pred[0], bbox_pred[1]), 
                                  bbox_pred[2] - bbox_pred[0], 
                                  bbox_pred[3] - bbox_pred[1], 
                                  fill=False, edgecolor='red', linewidth=2, label='Pred Box')
        ax.add_patch(rect_pred)
        
        if row_idx == 0:
            ax.legend(loc='upper right', fontsize='small')
            
        ax.axis('off')
        
        # --- Column 2: Ground Truth Mask ---
        ax = axes[row_idx, 1] if num_display > 1 else axes[1]
        ax.imshow(mask_true, cmap='gray')
        ax.set_title("Ground Truth Mask", fontsize=10)
        ax.axis('off')
        
        # --- Column 3: Predicted Mask ---
        ax = axes[row_idx, 2] if num_display > 1 else axes[2]
        ax.imshow(mask_pred, cmap='gray')
        ax.set_title("Predicted Mask", fontsize=10)
        ax.axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(args.output_dir, 'visualisation_best_worst.png')
    plt.savefig(save_path, dpi=150)
    print(f"Visualization saved to {save_path}")


if __name__ == '__main__':
    visualize_predictions()