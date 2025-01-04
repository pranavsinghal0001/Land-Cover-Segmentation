import os
import cv2
import math
import torch
import numpy as np
import segmentation_models_pytorch as smp
import streamlit as st
from tqdm import tqdm
from patchify import patchify, unpatchify
from utils.constants import Constants
from utils.plot import visualize
from utils.logger import custom_logger
from utils.root_config import get_root_config
from PIL import Image

# Define the password
APP_PASSWORD = "securepassword"  # Replace with your desired password

# Streamlit interface setup
st.title("Land Cover Semantic Segmentation")
st.write("Upload an image to perform land cover segmentation.")

# Password protection
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    password = st.text_input("Enter the password:", type="password")
    if password == APP_PASSWORD:
        st.session_state.authenticated = True
        st.success("Authentication successful!")
    else:
        if password:  # Avoid showing this message on initial load
            st.error("Incorrect password. Please try again.")
        st.stop()

# Configuration setup
ROOT, slice_config = get_root_config(__file__, Constants)

# Loading configuration
log_level = slice_config['vars']['log_level']
patch_size = slice_config['vars']['patch_size']
encoder = slice_config['vars']['encoder']
encoder_weights = slice_config['vars']['encoder_weights']
classes = slice_config['vars']['test_classes']  # Ensure this matches your classes
device = slice_config['vars']['device']

# Loading model
model_path = ROOT / slice_config['dirs']['model_dir'] / slice_config['vars']['model_name']
model = torch.load(model_path.as_posix(), map_location=torch.device(device))

# Model preprocessing function
preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
class_values = [Constants.CLASSES.value.index(cls.lower()) for cls in classes]

# User uploads the image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Reading and processing the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Padding image
    pad_height = (math.ceil(image.shape[0] / patch_size) * patch_size) - image.shape[0]
    pad_width = (math.ceil(image.shape[1] / patch_size) * patch_size) - image.shape[1]
    padded_shape = ((0, pad_height), (0, pad_width), (0, 0))
    image_padded = np.pad(image, padded_shape, mode='reflect')

    # Patchify the image
    patches = patchify(image_padded, (patch_size, patch_size, 3), step=patch_size//2)[:, :, 0, :, :, :]
    mask_patches = np.empty(patches.shape[:-1], dtype=patches.dtype)

    # Predict with model
    for i in tqdm(range(0, patches.shape[0])):
        for j in range(0, patches.shape[1]):
            img_patch = preprocessing_fn(patches[i, j, :, :, :])
            img_patch = img_patch.transpose(2, 0, 1).astype('float32')
            x_tensor = torch.from_numpy(img_patch).to(device).unsqueeze(0)
            pred_mask = model.predict(x_tensor)
            pred_mask = pred_mask.squeeze().cpu().numpy().round()
            pred_mask = pred_mask.transpose(1, 2, 0)
            pred_mask = pred_mask.argmax(2)
            mask_patches[i, j, :, :] = pred_mask

    # Unpatch the image
    pred_mask = unpatchify(mask_patches, image_padded.shape[:-1])
    pred_mask = pred_mask[:image.shape[0], :image.shape[1]]

    # Define a color map for visualization
    unique_classes = np.unique(pred_mask)
    class_colors = {
        idx: [np.random.randint(0, 256) for _ in range(3)] for idx in unique_classes
    }

    # Map the predicted class indices to RGB values
    colored_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in class_colors.items():
        colored_mask[pred_mask == class_idx] = color

    # Convert the colored mask to an image
    colored_mask_image = Image.fromarray(colored_mask)

    # Display the predicted mask
    st.image(colored_mask_image, caption="Predicted Mask", use_container_width=True)

    # Optionally display the color map legend
    st.write("Legend:")
    for class_idx, color in class_colors.items():
        st.markdown(f"**Class {class_idx}**: {color}")