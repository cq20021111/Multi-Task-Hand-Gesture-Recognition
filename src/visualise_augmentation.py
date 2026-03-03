import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from dataloader import HandGestureDataset

def inverse_normalize(tensor):
    """
    Inverse normalize the image tensor for visualization.
    Args:
        tensor (torch.Tensor): (3, H, W) normalized image
    Returns:
        numpy.ndarray: (H, W, 3) image in range [0, 1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    img = tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()

def visualize_augmentation():
    # --- 1. Set paths ---
    src_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(src_dir)
    data_dir = os.path.join(base_dir, 'data')
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return

    # --- 2. Initialize dataset (enable augmentation) ---
    # We explicitly pass augment=True to see the augmentation effects
    dataset = HandGestureDataset(data_dir, split='training', augment=True)
    
    # Select a sample index for visualization
    sample_idx = 0 
    num_variations = 5  # How many transformed variations to show
    
    print(f"Visualizing augmentations for sample index {sample_idx}...")
    
    # --- 3. Get multiple augmented results and plot ---
    fig, axes = plt.subplots(num_variations, 3, figsize=(12, 4 * num_variations))
    plt.subplots_adjust(hspace=0.3)
    
    # Set column titles
    cols = ['Augmented RGB', 'Augmented Depth', 'Augmented Mask']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for i in range(num_variations):
        # Each call to __getitem__ triggers random augmentation
        rgbd_image, mask, bbox, label = dataset[sample_idx]
        
        # Separate RGB and Depth
        rgb_tensor = rgbd_image[:3, :, :]
        depth_tensor = rgbd_image[3, :, :]
        
        # Inverse normalize RGB
        rgb_img = inverse_normalize(rgb_tensor)
        
        # Process Depth (already in [0, 1])
        depth_img = depth_tensor.numpy()
        
        # Process Mask (0 or 1)
        mask_img = mask.squeeze().numpy()
        
        # Plot RGB
        axes[i, 0].imshow(rgb_img)
        axes[i, 0].axis('off')
        
        # Plot Depth (using grayscale)
        axes[i, 1].imshow(depth_img, cmap='gray')
        axes[i, 1].axis('off')
        
        # Plot Mask (using grayscale)
        axes[i, 2].imshow(mask_img, cmap='gray')
        axes[i, 2].axis('off')
        
        # Draw bounding box on RGB image (optional)
        # bbox is [x_min, y_min, x_max, y_max] based on 224x224
        x_min, y_min, x_max, y_max = bbox.numpy()
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                             fill=False, edgecolor='red', linewidth=2)
        axes[i, 0].add_patch(rect)

    # Save results
    save_path = os.path.join(base_dir, 'results', 'augmentation_viz.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    # plt.show() # Uncomment if running in an environment with GUI support

if __name__ == '__main__':
    visualize_augmentation()