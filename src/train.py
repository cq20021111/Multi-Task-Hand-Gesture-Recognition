import torch
import torch.optim as optim
# from torch.utils.data import DataLoader
from model import MultiTaskHandNet
from dataloader import get_dataloader
from utils import calculate_iou_box, calculate_iou_mask
import argparse
import os

# Custom CIoU Loss
class CIoULoss(torch.nn.Module):
    def __init__(self):
        super(CIoULoss, self).__init__()

    def forward(self, preds, targets):
        """
        Calculate 1 - CIoU as the loss.
        Assume both preds and targets are in [xmin, ymin, xmax, ymax] format and normalized to [0, 1]
        """
        # Ensure coordinates are within bounds
        preds = torch.clamp(preds, min=0.0, max=1.0)
        
        # 1. Calculate IoU
        inter_xmin = torch.max(preds[:, 0], targets[:, 0])
        inter_ymin = torch.max(preds[:, 1], targets[:, 1])
        inter_xmax = torch.min(preds[:, 2], targets[:, 2])
        inter_ymax = torch.min(preds[:, 3], targets[:, 3])
        
        inter_w = torch.clamp(inter_xmax - inter_xmin, min=0.0)
        inter_h = torch.clamp(inter_ymax - inter_ymin, min=0.0)
        inter_area = inter_w * inter_h
        
        pred_w = preds[:, 2] - preds[:, 0]
        pred_h = preds[:, 3] - preds[:, 1]
        target_w = targets[:, 2] - targets[:, 0]
        target_h = targets[:, 3] - targets[:, 1]
        
        union_area = pred_w * pred_h + target_w * target_h - inter_area
        iou = inter_area / (union_area + 1e-6)
        
        # 2. Calculate squared distance between center points
        pred_center_x = (preds[:, 0] + preds[:, 2]) / 2
        pred_center_y = (preds[:, 1] + preds[:, 3]) / 2
        target_center_x = (targets[:, 0] + targets[:, 2]) / 2
        target_center_y = (targets[:, 1] + targets[:, 3]) / 2
        center_distance_sq = (pred_center_x - target_center_x)**2 + (pred_center_y - target_center_y)**2
        
        # 3. Calculate the squared diagonal of the smallest enclosing box (C)
        c_xmin = torch.min(preds[:, 0], targets[:, 0])
        c_ymin = torch.min(preds[:, 1], targets[:, 1])
        c_xmax = torch.max(preds[:, 2], targets[:, 2])
        c_ymax = torch.max(preds[:, 3], targets[:, 3])
        c_diagonal_sq = (c_xmax - c_xmin)**2 + (c_ymax - c_ymin)**2 + 1e-6
        
        # 4. Calculate aspect ratio consistency term (v) and alpha
        # 4/(pi^2) * (arctan(w_gt/h_gt) - arctan(w/h))^2
        v = (4 / (torch.pi ** 2)) * torch.pow(
            torch.atan(target_w / (target_h + 1e-6)) - torch.atan(pred_w / (pred_h + 1e-6)), 2
        )
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-6)
            
        # 5. Combine CIoU
        ciou = iou - (center_distance_sq / c_diagonal_sq) - alpha * v
        
        loss = 1.0 - ciou.mean()
        return loss

# Use a Config class instead of command line arguments for easier running in an IDE
class Config:
    # Get the directory where the current script is located (src)
    _src_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (dataset_and_code)
    _base_dir = os.path.dirname(_src_dir)
    
    # Deduce data directory and weights save directory
    data_dir = os.path.join(_base_dir, 'data')
    save_dir = os.path.join(_base_dir, 'weights')
    
    # Training hyperparameters
    epochs = 200         # Total number of training epochs
    batch_size = 32      # Batch size
    lr = 0.0012          # Learning rate
    w_box = 2.0          # Bounding box loss weight (reduced from 50 to 10/2 due to different scale of IoU Loss)
    w_mask = 1.0         # Segmentation mask loss weight
    val_split = 0.13     # Validation set ratio (13%)
    
    # Early Stopping parameters
    patience = 8         # Number of epochs to tolerate no decrease in validation loss
    min_delta = 0.0005   # Minimum threshold for validation loss decrease
    
    # Resume training flag
    resume = False       # Set to True to load a previously saved model to continue training (Note: set to False if model structure changed)
    resume_path = os.path.join(save_dir, 'best_model.pth') # Path of the model to load

def validate(model, val_loader, device, criterion_cls, criterion_box, criterion_mask, args):
    """
    Evaluate model performance on the validation set.
    """
    model.eval() # Switch to evaluation mode
    val_loss = 0.0
    val_cls_loss = 0.0
    val_box_loss = 0.0
    val_mask_loss = 0.0
    
    with torch.no_grad(): # Disable gradient calculation
        for images, masks, bboxes, labels in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            bboxes = bboxes.to(device)
            labels = labels.to(device)
            
            bboxes_norm = bboxes / 224.0
            
            cls_logits, bbox_preds, mask_logits = model(images)
            
            loss_cls = criterion_cls(cls_logits, labels)
            loss_box = criterion_box(bbox_preds, bboxes_norm)
            loss_mask = criterion_mask(mask_logits, masks)
            
            loss = loss_cls + args.w_box * loss_box + args.w_mask * loss_mask
            
            val_loss += loss.item()
            val_cls_loss += loss_cls.item()
            val_box_loss += loss_box.item()
            val_mask_loss += loss_mask.item()
            
    num_batches = len(val_loader)
    return (val_loss / num_batches, 
            val_cls_loss / num_batches, 
            val_box_loss / num_batches, 
            val_mask_loss / num_batches)

def train():
    """
    Main training function.
    Responsible for initializing the model, data loaders, optimizer, and executing the training loop.
    """
    args = Config()
    
    # Check if GPU is available, otherwise raise an error
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available.")
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # If the save directory does not exist, raise an error
    if not os.path.exists(args.save_dir):
        raise ValueError(f"Save directory {args.save_dir} does not exist. Please create it manually.")
    
    # --- 1. Load Data (including validation set split) ---
    print(f"Loading data from {args.data_dir}...")
    # Modified to receive two loaders
    train_loader, val_loader = get_dataloader(args.data_dir, split='training', batch_size=args.batch_size, val_split=args.val_split)
    print(f"Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # --- 2. Initialize Model ---
    model = MultiTaskHandNet(num_classes=10).to(device)
    
    # If resuming training is enabled, load weights
    if args.resume:
        if os.path.exists(args.resume_path):
            print(f"Resuming training from {args.resume_path}...")
            model.load_state_dict(torch.load(args.resume_path, map_location=device))
        else:
            print(f"Warning: Resume path {args.resume_path} does not exist. Starting from scratch.")
    
    # --- 3. Define Optimizer ---
    # Use Adam optimizer, add weight_decay to prevent overfitting (increased to 1e-3)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    
    # --- 3.1 Define Learning Rate Scheduler ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3)
    
    # --- 4. Define Loss Functions ---
    # Classification loss: CrossEntropyLoss (suitable for multi-class classification), add label_smoothing to prevent overfitting
    criterion_cls = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    # Bounding Box Regression Loss: Changed to CIoU Loss (comprehensively considers overlap area, center distance, and aspect ratio)
    criterion_box = CIoULoss()
    # Segmentation Loss: Binary Cross Entropy Loss (suitable for pixel-level binary classification)
    criterion_mask = torch.nn.BCEWithLogitsLoss()
    
    print("Starting training...")
    best_val_loss = float('inf') # Record the best validation set loss
    
    # Early Stopping counter
    early_stop_counter = 0
    
    for epoch in range(args.epochs):
        model.train() # Set to training mode
        
        # Record cumulative losses for each epoch
        running_loss = 0.0
        running_cls_loss = 0.0
        running_box_loss = 0.0
        running_mask_loss = 0.0
        
        for i, (images, masks, bboxes, labels) in enumerate(train_loader):
            # Move data to device (GPU/CPU)
            images = images.to(device)
            masks = masks.to(device)
            bboxes = bboxes.to(device)
            labels = labels.to(device)
            
            # Normalize bounding box coordinates to [0, 1] range, helps with training stability
            # Original coordinates are based on 224x224 image pixel coordinates
            bboxes_norm = bboxes / 224.0
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            cls_logits, bbox_preds, mask_logits = model(images)
            
            # Calculate respective losses
            loss_cls = criterion_cls(cls_logits, labels)
            loss_box = criterion_box(bbox_preds, bboxes_norm)
            loss_mask = criterion_mask(mask_logits, masks)
            
            # Total loss = weighted sum
            loss = loss_cls + args.w_box * loss_box + args.w_mask * loss_mask
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Accumulate statistics
            running_loss += loss.item()
            running_cls_loss += loss_cls.item()
            running_box_loss += loss_box.item()
            running_mask_loss += loss_mask.item()
            
            # Print logs every 10 batches
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], "
                      f"Train Loss: {loss.item():.4f} (Cls: {loss_cls.item():.4f}, Box: {loss_box.item():.4f}, Mask: {loss_mask.item():.4f})")
        
        # Calculate and print average training loss for the epoch
        avg_train_loss = running_loss / len(train_loader)
        
        # --- Validation Phase ---
        avg_val_loss, avg_val_cls, avg_val_box, avg_val_mask = validate(
            model, val_loader, device, criterion_cls, criterion_box, criterion_mask, args
        )
        
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} (Cls: {avg_val_cls:.4f}, Box: {avg_val_box:.4f}, Mask: {avg_val_mask:.4f})")
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")
        scheduler.step(avg_val_loss)
        
        # If validation loss is lower, save as the best model
        if avg_val_loss < (best_val_loss - args.min_delta):
            best_val_loss = avg_val_loss
            early_stop_counter = 0 # Reset counter
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f"New best model saved with Val Loss: {best_val_loss:.4f}")
        else:
            early_stop_counter += 1
            print(f"EarlyStopping counter: {early_stop_counter} out of {args.patience}")
            if early_stop_counter >= args.patience:
                print("Early stopping triggered.")
                break


if __name__ == '__main__':
    train()