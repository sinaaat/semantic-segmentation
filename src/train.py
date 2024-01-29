# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from pathlib import Path
from src.models.resnet_fcn import ResNetFCN


def label_to_tensor(label):
    return ToTensor()(label)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet mean and std
    ])

    # Load Cityscapes dataset
    data_dir = Path(__file__).parents[1] / 'data'
    train_dataset = Cityscapes(root=data_dir, split='train', mode='fine', target_type='semantic',
                               transform=transform, target_transform=label_to_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataset = Cityscapes(root=data_dir, split='val', mode='fine', target_type='semantic',
                             transform=transform, target_transform=label_to_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)

    # Initialize model
    model = ResNetFCN(num_classes=30).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    # Train model
    for epoch in range(num_epochs):
        print(epoch)
        model.train()
        running_loss = 0.0
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader):.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_images, val_labels in val_dataloader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)

                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')
    print('Finished Training')
    # save model
    torch.save(model.state_dict(), 'resnet_fcn_cityscapes.pth')

if __name__ == '__main__':
    main()
