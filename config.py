import os

# Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed audio formats
ALLOWED_EXTENSIONS = {"mp3", "wav", "mp4", "m4a"}

# HuggingFace Token (replace with your token)
HUGGINGFACE_TOKEN = "YOUR_HUGGINGFACE_TOKEN"
