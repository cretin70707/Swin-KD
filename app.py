import torch
import cv2
import numpy as np
import streamlit as st
import os
from datasets.EchoSet import loadvideo

# Function to calculate ejection fraction
import torch
import numpy as np
from datasets.EchoSet import loadvideo

def predict_ejection_fraction(model, video_path, padding=None, device='cpu'):
    # Load the video
    video = loadvideo(video_path).astype(np.float32)
    video /= 255.0  
    video = np.moveaxis(video, 0, 1)  

    # Add padding 
    if padding is not None:
        p = padding
        video = np.pad(video, ((0,0),(0,0),(p,p),(p,p)), mode='constant', constant_values=0)

    # Convert  to a tensor
    video_tensor = torch.from_numpy(video).unsqueeze(0).to(device)

    model.eval()

    # Forward pass
    with torch.no_grad():
        output = model(video_tensor)


    predicted_ejection_fraction = output.item() * 100  

    return predicted_ejection_fraction

# Function to save uploaded video
def save_uploaded_file(uploaded_file, folder_path):
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)  # Create folder if it doesn't exist

        # Check for .avi extension
        if uploaded_file.name.endswith(".avi"):
            with open(os.path.join(folder_path, 'input.avi'), "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Video '{uploaded_file.name}' saved successfully!")
        else:
            st.warning("Only .avi files are allowed. Please try again.")
    except Exception as e:
        st.error(f"Error saving video: {e}")

st.title('Calculate your ejection fraction!')
uploaded_file = st.file_uploader("Upload Your Echocardiogram", type="avi")
file_path = "uploaded_videos"

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kd_model = torch.load('distilled_model.pth')
kd_model.to(device)
video_path = "uploaded_videos/input.avi"
padding = None

if uploaded_file is not None:
    # Code to handle the uploaded video (place your code here)
    st.success(f"Video '{uploaded_file.name}' uploaded!")
    save_uploaded_file(uploaded_file, file_path)
    
if st.button("Calculate Ejection Fraction"):
    try:
        predicted_ef = predict_ejection_fraction(kd_model, video_path, padding, device)
        print(predicted_ef)
        st.write(f'Your predicted ejection fraction is {predicted_ef:.6f}%')
    except:
        st.error("An error occurred while predicting EF rate. Please try again.")    
           
