import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import json
import os
import time
import ssl

# Bypass SSL verification for model download
ssl._create_default_https_context = ssl._create_unverified_context

# Configuration
TRAIN_DIR = 'data/train'
VALID_DIR = 'data/valid'
MODEL_PATH = 'plant_disease_model.pth'
CLASSES_PATH = 'classes.json'
BATCH_SIZE = 32
EPOCHS = 1  # Keeping it low for demonstration/speed on CPU
IMG_SIZE = 224

class MultiCropDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        # Traverse directory: Root -> Crop -> Disease -> Images
        crops = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        idx = 0
        for crop in crops:
            crop_path = os.path.join(root_dir, crop)
            diseases = sorted([d for d in os.listdir(crop_path) if os.path.isdir(os.path.join(crop_path, d))])
            
            for disease in diseases:
                class_name = f"{crop} - {disease}"
                if class_name not in self.class_to_idx:
                    self.classes.append(class_name)
                    self.class_to_idx[class_name] = idx
                    idx += 1
                
                disease_path = os.path.join(crop_path, disease)
                for img_name in os.listdir(disease_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(disease_path, img_name), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = datasets.folder.default_loader(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        return image, target

def train():
    print("Setting up data transforms...")
    # Data augmentation and normalization for training
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Loading datasets...")
    image_datasets = {
        'train': MultiCropDataset(TRAIN_DIR, data_transforms['train']),
        'valid': MultiCropDataset(VALID_DIR, data_transforms['valid'])
    }
    
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    }
    
    class_names = image_datasets['train'].classes
    print(f"Classes found: {len(class_names)}")
    
    # Save classes
    with open(CLASSES_PATH, 'w') as f:
        json.dump(class_names, f)
    print(f"Classes saved to {CLASSES_PATH}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Pretrained MobileNetV2
    print("Loading MobileNetV2...")
    model = models.mobilenet_v2(pretrained=True)
    
    # Freeze weights
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace classifier
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # Training Loop
    print("Starting training...")
    since = time.time()

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch}/{EPOCHS - 1}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            # Limit batches for speed if needed, but let's try full first
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                if i > 50: break # Limit for demonstration speed on CPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if i % 10 == 0:
                    print(f"Batch {i} Loss: {loss.item():.4f}")

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # Save Model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
