import os
import io
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Set up Streamlit page
st.set_page_config(page_title="RoomCraft", page_icon=":house:")
st.title("RoomCraft: Interior Design Assistant :house:")

# Replace Hugging Face with Groq API token
GROQ_API_TOKEN = os.getenv('GROQ_API_TOKEN')  # Make sure your token is stored in an environment variable

# Update this with the correct Groq API endpoint (e.g., "/v1/image-generation")
GROQ_API_URL = "https://api.groq.com/v1/image-generation"  # Hypothetical, replace with correct endpoint

# Function to generate image using Groq
def generate_image(prompt):
    headers = {
        'Authorization': f'Bearer {GROQ_API_TOKEN}',
        'Content-Type': 'application/json',
    }
    data = {"prompt": prompt}

    response = requests.post(GROQ_API_URL, json=data, headers=headers)
    if response.status_code == 200:
        image_bytes = response.content  # Assuming the API returns image bytes
        return Image.open(io.BytesIO(image_bytes))
    else:
        raise Exception(f"Error from Groq API: {response.status_code}, {response.text}")

# Function to process uploaded image and user preferences
def process_image(uploaded_image, style, room_type):
    img = Image.open(uploaded_image)
    prompt = f"Transform this {room_type} into a {style} style. {style} interior design for {room_type}."
    return generate_image(prompt)

# Main application
def main():
    st.sidebar.header("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image of your room", type=["jpg", "jpeg", "png"])

    st.sidebar.header("Design Preferences")
    style = st.sidebar.selectbox("Choose a style", ["Modern", "Vintage", "Minimalist", "Industrial", "Bohemian"])
    room_type = st.sidebar.selectbox("Room type", ["Living Room", "Bedroom", "Kitchen", "Bathroom", "Office"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("Generate Design"):
            with st.spinner("Generating new design..."):
                try:
                    new_image = process_image(uploaded_file, style, room_type)
                    st.image(new_image, caption="Generated Design", use_column_width=True)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload an image to start.")

if __name__ == "__main__":
    main()
