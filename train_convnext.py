# train_convnext.py
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ====================== CONFIG ======================
CSV_PATH = "dataset.csv"
IMG_SIZE = 512
BATCH_SIZE = 12          # sicuro per 512x512 + GradCAM su RTX 5070 (12-16GB)
NUM_EPOCHS = 30
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 42

LABEL_MAP = {"normale": 0, "istotipo1": 1, "istotipo2": 2, "istotipo3": 3, "istotipo4": 4, "istotipo5": 5}
NUM_CLASSES = len(LABEL_MAP)
CLASS_NAMES = list(LABEL_MAP.keys())

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ====================== OTTIMIZZAZIONI RTX 5070 ======================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ====================== TRANSFORMS ======================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(90),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ====================== DATASET ======================
class HistologyDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            img = self.transform(img)
        return img, label, row["case_id"], row["image_path"]

# ====================== MODELLO ======================
def get_model():
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = convnext_tiny(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)
    return model.to(DEVICE)

# ====================== TRAIN / EVAL ======================
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels, _, _ in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_paths = [], [], []
    for images, labels, _, paths in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_paths.extend(paths)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return running_loss / len(loader.dataset), bal_acc, all_preds, all_labels, all_paths

# ====================== GRAD-CAM ======================
def generate_gradcam_batch(model, images, target_classes, original_paths, save_dir):
    model.eval()
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(images.size(0)):
        img_tensor = images[i].unsqueeze(0)
        targets = [ClassifierOutputTarget(int(target_classes[i]))]
        
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
        
        # De-normalize
        img_np = images[i].cpu().permute(1, 2, 0).numpy()
        img_np = img_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        img_np = np.clip(img_np, 0, 1)
        
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        fname = Path(original_paths[i]).stem + "_gradcam.jpg"
        plt.imsave(save_dir / fname, visualization)

# ====================== MAIN ======================
def main():
    df = pd.read_csv(CSV_PATH)
    df["label"] = df["label"].map(LABEL_MAP)
    
    case_df = df.groupby("case_id")["label"].first().reset_index()
    case_ids = case_df["case_id"].values
    case_labels = case_df["label"].values
    
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    
    fold_results = []
    all_true, all_pred = [], []
    
    for fold, (train_idx, test_idx) in enumerate(rskf.split(case_ids, case_labels)):
        print(f"\n===== FOLD {fold+1}/{N_SPLITS*N_REPEATS} =====")
        
        train_cases = case_ids[train_idx]
        test_cases = case_ids[test_idx]
        
        train_df = df[df["case_id"].isin(train_cases)]
        test_df = df[df["case_id"].isin(test_cases)]
        
        train_loader = DataLoader(HistologyDataset(train_df, train_transform),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
        test_loader = DataLoader(HistologyDataset(test_df, val_transform),
                                 batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
        
        model = get_model()
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        scaler = GradScaler()
        
        best_bal_acc = 0.0
        best_state = None
        
        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
            val_loss, val_bal_acc, _, _, _ = evaluate(model, test_loader, criterion)
            scheduler.step()
            
            if val_bal_acc > best_bal_acc:
                best_bal_acc = val_bal_acc
                best_state = model.state_dict()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val BalAcc: {val_bal_acc:.4f}")
        
        model.load_state_dict(best_state)
        torch.save(best_state, OUTPUT_DIR / f"best_model_fold{fold+1}.pth")
        
        _, final_bal_acc, preds, labels, paths = evaluate(model, test_loader, criterion)
        print(f">>> Fold {fold+1} Best BalAcc: {final_bal_acc:.4f}")
        
        # Grad-CAM sulle immagini di test del fold
        print("Generazione Grad-CAM...")
        model.eval()
        for images, labels_batch, _, paths_batch in test_loader:
            images = images.to(DEVICE)
            generate_gradcam_batch(model, images, labels_batch, paths_batch,
                                   save_dir=OUTPUT_DIR / f"gradcam_fold{fold+1}")
            break  # solo primo batch per non saturare (modifica se vuoi tutte)
        
        fold_results.append(final_bal_acc)
        all_true.extend(labels)
        all_pred.extend(preds)
    
    # Risultati finali
    print("\n===== RISULTATI FINALI =====")
    print(f"Balanced Accuracy: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    
    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix - Tutti i fold")
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()
    
    print(classification_report(all_true, all_pred, target_names=CLASS_NAMES, digits=4))

if __name__ == "__main__":
    main()