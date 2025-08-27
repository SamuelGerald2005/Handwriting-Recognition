import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin"

# === 🚩 SET THESE ===
train_csv_path = r"C:/Users/sam20/PycharmProjects/Handwriting_Recognition/Updated Labels/GroundTruth-TrainSplit-AfterSplit.csv"
val_csv_path = r"C:/Users/sam20/PycharmProjects/Handwriting_Recognition/Updated Labels/GroundTruth-ValSplit.csv"
test_csv_path = r"C:/Users/sam20/PycharmProjects/Handwriting_Recognition/Updated Labels/GroundTruth-Test.csv"
label_map_path = r"C:/Users/sam20/PycharmProjects/Handwriting_Recognition/Updated Labels/label_map_with_chars.csv"
train_img_dir = r"C:/Users/sam20/PycharmProjects/Handwriting_Recognition/Preprocessed_Letters/70-30-split/Train"
test_img_dir = r"C:/Users/sam20/PycharmProjects/Handwriting_Recognition/Preprocessed_Letters/70-30-split/Test"

# === ⚙️ SETTINGS ===
img_size = (64, 64)
batch_size = 32
epochs = 200
num_classes = 156
patience = 10


# === DEVICE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === TRANSFORMS ===
train_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.Grayscale(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(
        degrees=15,               # Rotation ±15°
        scale=(0.8, 1.2),         # Zoom in/out ±20%
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_test_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === DATASET ===
class HandwritingDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['FileNames'])
        label = row['Ground Truth']

        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        return image, label

# === MODEL ===
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 64x64 → 64x64
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 32x32

            # Block 2: 32x32 → 32x32
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 16x16

            # Block 3: 16x16 → 16x16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 8x8

            # Block 4: 8x8 → 8x8
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 4x4

            # Block 5: 4x4 → 4x4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 2x2
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),              # 512 * 2 * 2 = 2048
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



# === PLOTTING ===
def plot_loss_curve(train_losses, val_losses):
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss", marker='x')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy(train_accs, val_accs):
    plt.figure()
    plt.plot(range(1, len(train_accs)+1), train_accs, label='Train Accuracy', marker='o')
    plt.plot(range(1, len(val_accs)+1), val_accs, label='Validation Accuracy', marker='x')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_top_confused(cm, labels, top_n=10):
    cm_offdiag = cm.copy()
    np.fill_diagonal(cm_offdiag, 0)
    top_idx = np.argsort(cm_offdiag.sum(axis=1))[-top_n:]

    cm_top = cm[np.ix_(top_idx, top_idx)]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_top, annot=True, fmt='d', cmap="Blues",
                xticklabels=[labels[i] for i in top_idx],
                yticklabels=[labels[i] for i in top_idx])
    plt.title(f"Top {top_n} Confused Classes")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

# === TRAINING ===
def train_model():
    train_dataset = HandwritingDataset(train_csv_path, train_img_dir, transform=train_transform)
    val_dataset = HandwritingDataset(val_csv_path, train_img_dir, transform=val_test_transform)
    test_dataset = HandwritingDataset(test_csv_path, test_img_dir, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = CNNModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params}")

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    epochs_no_improve = 0



    # Load best model before evaluation
    model.load_state_dict(torch.load("bestcnn_model_scheduler.pth"))

    # Plot training curves
    plot_loss_curve(train_losses, val_losses)
    plot_accuracy(train_accuracies, val_accuracies)

    # Final Evaluation on Test Set (with Confidence)
    model.eval()
    all_preds = []
    all_labels = []
    all_confs = []  # ⬅️ New: store confidence scores
    features_list = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)  # logits
            probs = F.softmax(outputs, dim=1)  # get probabilities
            confs, preds = torch.max(probs, dim=1)  # top-1 prediction + confidence

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confs.extend(confs.cpu().numpy())  # ⬅️ Save confidence scores
            features_list.append(outputs.cpu().numpy())

    features_array = np.vstack(features_list)
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n🌟 Test Accuracy: {acc:.4f}")
    print("\n📾 Classification Report:")
    print(classification_report(all_labels, all_preds))

    print("\n🔍 Sample Predictions with Confidence Scores:")
    for i in range(10):  # change 10 to how many samples you want to view
        print(f"Sample {i}: True = {all_labels[i]}, Predicted = {all_preds[i]}, Confidence = {all_confs[i]:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    label_map = pd.read_csv(label_map_path)
    labels = label_map['class'].tolist()
    plot_top_confused(cm, labels, top_n=10)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    train_model()
