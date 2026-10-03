# app.py
import streamlit as st
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchvision.models import convnext_tiny
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

st.set_page_config(page_title="Histology Classifier - GradCAM", layout="wide")
st.title("Classificazione Istologica Polmonare Canina + Grad-CAM")

# Config
IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["normale", "istotipo1", "istotipo2", "istotipo3", "istotipo4", "istotipo5"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

@st.cache_resource
def load_model(model_path):
    model = convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def preprocess(img):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf(img).unsqueeze(0)

def run_gradcam(model, img_tensor, pred_class):
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
    
    img_np = img_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    img_np = img_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img_np = np.clip(img_np, 0, 1)
    
    return show_cam_on_image(img_np, grayscale_cam, use_rgb=True), img_np

# Sidebar
st.sidebar.header("Impostazioni")
model_path = st.sidebar.text_input("Percorso modello (.pth)", "results/best_model_fold1.pth")
uploaded_file = st.sidebar.file_uploader("Carica immagine istologica", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file and Path(model_path).exists():
    model = load_model(model_path)
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Immagine originale")
        st.image(image, use_container_width=True)
    
    img_tensor = preprocess(image).to(DEVICE)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_class = probs.argmax().item()
    
    st.subheader("Predizione")
    st.write(f"**Classe predetta:** {CLASS_NAMES[pred_class]}")
    st.write("Probabilità:")
    for i, p in enumerate(probs):
        st.progress(float(p), text=f"{CLASS_NAMES[i]}: {p:.3f}")
    
    if st.button("Genera Grad-CAM"):
        with st.spinner("Calcolo Grad-CAM..."):
            cam_img, orig_np = run_gradcam(model, img_tensor, pred_class)
        
        with col2:
            st.subheader("Grad-CAM")
            st.image(cam_img, use_container_width=True)
else:
    st.info("Carica un'immagine e specifica il percorso di un modello addestrato.")