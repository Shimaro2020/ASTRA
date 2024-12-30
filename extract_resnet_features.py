"""
Dataset/
├── Train/
│   ├── Class1/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   ├── Class2/
│   │   ├── video1.mp4
│   │   └── ...
└── Test/
    ├── Class1/
    │   ├── video1.mp4
    │   └── ...
    ├── Class2/
    │   ├── video1.mp4
    │   └── ...

"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import cv2
from tqdm import tqdm

# Device
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the pretrained model
resnet152 = models.resnet152(pretrained=True)
resnet152.eval()
resnet152.to(DEVICE)

# Replace the fc layer with an identity layer to extract features
class Identity(nn.Module):
    def forward(self, input_):
        return input_

resnet152.fc = Identity()

# Image transformations
transf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_frames(video_path):
    """
    Extract frames from a video file and apply transformations.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transf(frame)
        frames.append(frame)
        success, frame = cap.read()
    cap.release()
    return torch.stack(frames) if frames else None

def frames_features(frames):
    """
    Extract features from frames using ResNet152.
    """
    frames = frames.to(DEVICE)
    output = []
    batch_size = 10  # Process in batches
    for start_index in range(0, len(frames), batch_size):
        end_index = min(start_index + batch_size, len(frames))
        frame_batch = frames[start_index:end_index]
        avg_pool_value = resnet152(frame_batch)
        output.append(avg_pool_value.detach().cpu().numpy())
    return np.concatenate(output, axis=0)

def process_dataset(dataset_folder, save_folder):
    """
    Process all videos in Train and Test sets and save extracted features
    in the same folder structure as the dataset.
    
    """
    for split in ['Train', 'Test']: #change this to accomodate your dataset Sets, eg. Train, Val, Test
        split_folder = os.path.join(dataset_folder, split)
        save_split_folder = os.path.join(save_folder, split)

        if not os.path.exists(save_split_folder):
            os.makedirs(save_split_folder)

        class_labels = [d for d in os.listdir(split_folder) if os.path.isdir(os.path.join(split_folder, d))]

        for label in class_labels:
            label_folder = os.path.join(split_folder, label)
            save_label_folder = os.path.join(save_split_folder, label)

            if not os.path.exists(save_label_folder):
                os.makedirs(save_label_folder)

            video_files = [f for f in os.listdir(label_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

            for video_file in tqdm(video_files, desc=f"Processing {split}/{label}", ncols=100):
                video_path = os.path.join(label_folder, video_file)
                save_path = os.path.join(save_label_folder, os.path.splitext(video_file)[0] + '.npy')

                print(f"Processing {split}/{label}/{video_file}...")
                frames = get_frames(video_path)
                if frames is None:
                    print(f"Skipping {split}/{label}/{video_file}: No frames extracted.")
                    continue

                features = frames_features(frames)
                np.save(save_path, features)  # Save features as .npy file
                print(f"Saved features for {split}/{label}/{video_file} to {save_path}. Shape: {features.shape}")

def main():
    dataset_folder = "./Dataset"  # Path to the dataset folder containing Train and Test sets
    save_folder = "./Features"   # Path to save extracted features
        
    process_dataset(dataset_folder, save_folder)
    print("Feature extraction and saving complete.")

if __name__ == '__main__':
    main()
