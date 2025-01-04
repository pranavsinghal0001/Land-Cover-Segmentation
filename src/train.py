import os
import shutil
import torch
import splitfolders
import streamlit as st
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from utils.constants import Constants
from utils.logger import custom_logger
from utils.root_config import get_root_config
from utils.patching import patching, discard_useless_patches
from utils.preprocess import get_training_augmentation, get_preprocessing
from utils.dataset import SegmentationDataset

# Streamlit interface setup
st.title("Land Cover Semantic Segmentation Training")
st.write("Configure your training parameters and start training.")

# Loading configuration
ROOT, slice_config = get_root_config(__file__, Constants)

# Configuration setup
log_level = slice_config['vars']['log_level']
patch_size = slice_config['vars']['patch_size']
batch_size = slice_config['vars']['batch_size']
model_arch = slice_config['vars']['model_arch']
encoder = slice_config['vars']['encoder']
encoder_weights = slice_config['vars']['encoder_weights']
activation = slice_config['vars']['activation']
optimizer_choice = slice_config['vars']['optimizer_choice']
init_lr = slice_config['vars']['init_lr']
epochs = slice_config['vars']['epochs']
classes = slice_config['vars']['train_classes']
device = slice_config['vars']['device']

# Get user input for training configuration
patch_size = st.slider("Patch Size", min_value=32, max_value=512, step=32, value=patch_size)
batch_size = st.slider("Batch Size", min_value=4, max_value=32, step=4, value=batch_size)
epochs = st.number_input("Number of Epochs", min_value=1, max_value=100, value=epochs)

# Training data directories
train_dir = ROOT / slice_config['dirs']['data_dir'] / slice_config['dirs']['train_dir']
train_dir = train_dir.as_posix()

img_dir = ROOT / slice_config['dirs']['data_dir'] / slice_config['dirs']['train_dir'] / slice_config['dirs']['image_dir']
img_dir = img_dir.as_posix()

mask_dir = ROOT / slice_config['dirs']['data_dir'] / slice_config['dirs']['train_dir'] / slice_config['dirs']['mask_dir']
mask_dir = mask_dir.as_posix()

# Training Button
if st.button("Start Training"):
    try:
        # Initialize the logger
        log_dir = ROOT / slice_config['dirs']['log_dir']
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / slice_config['vars']['train_log_name']
        log_path = log_path.as_posix()
        logger = custom_logger("Land Cover Semantic Segmentation Train Logs", log_path, log_level)

        # Start training
        st.write("Training started...")

        # Create the model
        smp_model = getattr(smp, model_arch)
        model = smp_model(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=len(classes),
            activation=activation,
        )

        # Prepare dataset and dataloaders
        train_dataset = SegmentationDataset(
        img_dir,
        mask_dir,
        classes=classes,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(get_preprocessing(encoder, encoder_weights)),  # Correct function call
)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Start the training loop (logging and feedback using Streamlit)
        for epoch in range(epochs):
            st.write(f"Epoch {epoch+1}/{epochs} - Training...")
            # Add actual training steps here and display live progress (metrics, etc.)
           
            # Optionally save models during training
            if epoch % 5 == 0:
                model_save_path = f"model_epoch_{epoch+1}.pth"
                torch.save(model.state_dict(), model_save_path)
                st.write(f"Model saved at epoch {epoch+1}!")

    except Exception as e:
        st.error(f"Error occurred: {e}")
