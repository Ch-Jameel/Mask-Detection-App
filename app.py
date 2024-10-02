import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Function to load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# Load the custom CSS
load_css('style.css')


# Load your trained YOLO model
model = YOLO('best_model.pt') 


# Load the logo image
logo = Image.open("Dataset\jameel_logo.jpeg")
# # # Add logo to the sidebar
st.sidebar.image(logo, use_column_width=True)


# veriable to track the uploaded image
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file=None
# Upload an image
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
st.title("Mask Detection APP")


 

# Store the uploaded image in session state
if uploaded_file is not None:
    st.session_state.uploaded_file = Image.open(uploaded_file)

if st.sidebar.button("Predict"):
        image = Image.open(uploaded_file)      # Open the uploaded image
        image = image.convert("RGB")        # Convert image to RGB (remove alpha channel if present)
        
        # Make predictions using the model
        results = model.predict(np.array(image))

        # Display prediction results on the original image
        if results:
            # Show the predictions directly without saving
            for result in results:
                st.image(result.plot(), caption='Prediction Result', use_column_width=True)  # Display the prediction result
        else:
            st.write("No predictions were made.")

        st.session_state.uploaded_file = None  # Clear the image from session state

# Display the uploaded image initially
if st.session_state.uploaded_file is not None:
    uploaded_image = st.session_state.uploaded_file
    resized_image = uploaded_image.resize((1200,1200))  # Change size as needed
    st.image(resized_image, caption="Uploaded Image", use_column_width=True)
