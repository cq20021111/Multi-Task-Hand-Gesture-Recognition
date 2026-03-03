import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2

class HandGestureDataset(Dataset):
    """
    Custom hand gesture dataset class for loading RGB images, depth images, and corresponding segmentation masks.
    Also handles data preprocessing, augmentation, and extracting bounding boxes from masks.
    """
    def __init__(self, root_dir, split='training', transform=None, augment=None):
        """
        Initialize the dataset.
        Args:
            root_dir (string): Root directory of the dataset.
            split (string): 'training' or 'test'.
            transform (callable, optional): Optional transform to be applied (mainly handled manually in __getitem__).
            augment (bool, optional): Whether to apply data augmentation. Defaults to True if split='training'.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        if augment is None:
            self.augment = (split == 'training')
        else:
            self.augment = augment

        # Define Albumentations augmentation pipeline
        if self.augment:
            self.aug_pipeline = A.Compose([
                A.Rotate(limit=15, p=0.5, border_mode=0, value=0), # Random rotation +/- 15 degrees, fill with black (value=0)
                A.RandomScale(scale_limit=0.2, p=0.5), # Random scale +/- 20%
                A.HorizontalFlip(p=0.5), # Horizontal flip
                A.Perspective(scale=(0.05, 0.1), p=0.3, pad_mode=0, fit_output=True), # Perspective transform, fill with black
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3), # Color jitter
                # A.GaussNoise(var_limit=(10.0, 50.0), p=0.2), # Gaussian noise (temporarily removed, might cause dimension issues)
                A.MotionBlur(blur_limit=3, p=0.2), # Motion blur
            ], additional_targets={'depth': 'image', 'mask': 'mask'}) # Ensure depth and mask are transformed synchronously
        else:
            self.aug_pipeline = None

        # Define 10 hand gesture class names
        self.classes = [
            'G01_call', 'G02_dislike', 'G03_like', 'G04_ok', 'G05_one',
            'G06_palm', 'G07_peace', 'G08_rock', 'G09_stop', 'G10_three'
        ]
        # Create class name -> index mapping dictionary
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []

        # Iterate through directory structure to load data
        data_path = os.path.join(root_dir, split)
        if not os.path.exists(data_path):
            raise ValueError(f"Directory {data_path} does not exist.")

        for cls_name in self.classes:
            cls_dir = os.path.join(data_path, cls_name)
            if not os.path.isdir(cls_dir):
                raise ValueError(f"Class directory {cls_dir} does not exist or is not a directory.")
            
            rgb_dir = os.path.join(cls_dir, 'rgb')
            depth_dir = os.path.join(cls_dir, 'depth')
            anno_dir = os.path.join(cls_dir, 'annotation')
            
            if not os.path.exists(rgb_dir):
                raise ValueError(f"RGB directory {rgb_dir} does not exist.")
            if not os.path.exists(depth_dir):
                raise ValueError(f"Depth directory {depth_dir} does not exist.")
            if not os.path.exists(anno_dir):
                raise ValueError(f"Annotation directory {anno_dir} does not exist.")

            # Assume filenames in rgb, depth, and annotation folders correspond one-to-one
            for img_name in os.listdir(rgb_dir):
                if img_name.endswith('.png'):
                    rgb_path = os.path.join(rgb_dir, img_name)
                    depth_path = os.path.join(depth_dir, img_name)
                    anno_path = os.path.join(anno_dir, img_name)
                    
                    # Only add to list if corresponding annotation and depth files exist
                    if os.path.exists(anno_path) and os.path.exists(depth_path):
                        self.samples.append({
                            'rgb_path': rgb_path,
                            'depth_path': depth_path,
                            'anno_path': anno_path,
                            'label': self.class_to_idx[cls_name]
                        })
                    else:
                        print(f"Warning: Annotation or Depth file not found for RGB image {rgb_path}.")

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get sample at the specified index.
        Returns:
            image: Preprocessed RGB-D image tensor (4, 224, 224). First 3 channels are RGB, 4th channel is Depth.
            mask_tensor: Binary segmentation mask tensor (1, 224, 224)
            bbox: Bounding box coordinates tensor [x_min, y_min, x_max, y_max]
            label: Class label index
        """
        sample_info = self.samples[idx]
        rgb_path = sample_info['rgb_path']
        depth_path = sample_info['depth_path']
        anno_path = sample_info['anno_path']
        label = sample_info['label']

        # Load images
        image = Image.open(rgb_path).convert('RGB')
        depth = Image.open(depth_path) # Usually single channel (L)
        mask = Image.open(anno_path) 
        
        # Convert to numpy arrays for Albumentations processing
        image_np = np.array(image)
        depth_np = np.array(depth)
        mask_np = np.array(mask)
        
        # Apply data augmentation
        if self.augment and self.aug_pipeline is not None:
            # Ensure data type is uint8
            if image_np.dtype != np.uint8:
                image_np = image_np.astype(np.uint8)
            if depth_np.dtype != np.uint8:
                depth_np = depth_np.astype(np.uint8)
            if mask_np.dtype != np.uint8:
                mask_np = mask_np.astype(np.uint8)

            
            # Check depth dimensions
            if len(depth_np.shape) == 3:
                depth_np = depth_np[:, :, 0] # Take the first channel
            
            # Check mask dimensions
            if len(mask_np.shape) == 3:
                mask_np = mask_np[:, :, 0] # Take the first channel


            # Ensure they are continuous arrays (C-contiguous)
            image_np = np.ascontiguousarray(image_np)
            depth_np = np.ascontiguousarray(depth_np)
            mask_np = np.ascontiguousarray(mask_np)

            transformed = self.aug_pipeline(image=image_np, depth=depth_np, mask=mask_np)
            image_np = transformed['image']
            depth_np = transformed['depth']
            mask_np = transformed['mask']

        # Convert back to PIL for torchvision's Resize and ToTensor
        image = Image.fromarray(image_np)
        depth = Image.fromarray(depth_np)
        mask = Image.fromarray(mask_np)

        # Uniformly resize images and masks to fixed size (224x224)
        target_size = (224, 224)
        image = TF.resize(image, target_size)
        depth = TF.resize(depth, target_size)
        # Mask uses nearest neighbor interpolation to ensure pixel values are not smoothed (kept at 0 or 255)
        mask = TF.resize(mask, target_size, interpolation=transforms.InterpolationMode.NEAREST)


        # Convert to Tensor
        image = TF.to_tensor(image) # Automatically normalizes to [0, 1]
        depth = TF.to_tensor(depth) # Automatically normalizes to [0, 1]
        
        # Process mask
        mask_np = np.array(mask)
        # If mask is RGB format (though usually single channel), take max channel or convert to grayscale
        if len(mask_np.shape) == 3:
            mask_np = np.max(mask_np, axis=2)
        
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0)
        mask_tensor = (mask_tensor > 0).float() # Binarize: pixels > 0 set to 1 (foreground), otherwise 0 (background)

        # RGB image normalization (using ImageNet mean and std)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Concatenate RGB and Depth into a 4-channel input (RGBD)
        # image: (3, 224, 224), depth: (1, 224, 224) -> rgbd: (4, 224, 224)
        rgbd_image = torch.cat([image, depth], dim=0)

        # Calculate bounding box from resized mask
        # Get coordinates of all foreground pixels
        y_indices, x_indices = torch.where(mask_tensor[0] > 0)
        
        if len(y_indices) == 0:
            # If mask is empty (no hand), return all-zero box
            bbox = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
        else:
            # Calculate minimum bounding rectangle
            x_min = torch.min(x_indices).float()
            x_max = torch.max(x_indices).float()
            y_min = torch.min(y_indices).float()
            y_max = torch.max(y_indices).float()
            bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

        return rgbd_image, mask_tensor, bbox, torch.tensor(label, dtype=torch.long)

def get_dataloader(root_dir, split='training', batch_size=32, shuffle=True, num_workers=6, val_split=0.0):
    """
    Create and return DataLoader instance.
    If val_split > 0.0, split dataset into train and validation sets, and return (train_loader, val_loader).
    Otherwise return a single dataloader.
    """
    dataset = HandGestureDataset(root_dir, split=split)
    
    if val_split > 0.0:
        # Get dataset size
        # Instantiate a temporary dataset object here to get the sample list
        full_dataset = HandGestureDataset(root_dir, split=split)
        dataset_size = len(full_dataset)
        val_size = int(dataset_size * val_split)
        train_size = dataset_size - val_size
        
        # Use Generator to fix seed, ensuring consistent split every time
        generator = torch.Generator().manual_seed(42)
        
        # Generate random indices
        indices = torch.randperm(dataset_size, generator=generator).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create two independent dataset objects
        # Train set: enable augmentation
        train_dataset = HandGestureDataset(root_dir, split=split, augment=True)
        # Validation set: disable augmentation (this is key, validation set should be clean)
        val_dataset = HandGestureDataset(root_dir, split=split, augment=False)
        
        # Use Subset to specify indices
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        val_subset = torch.utils.data.Subset(val_dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader
    else:
        # If not splitting validation set, decide whether to augment based on split (default logic)
        dataset = HandGestureDataset(root_dir, split=split)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return dataloader
