import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity

# Paths
UPLOAD_FOLDER = 'trial-main/Eigenfaces/static/uploads'
USER_FOLDER = 'trial-main/Eigenfaces/data/users'
MODEL_PATH = "trial-main/Eigenfaces/model/pca_model.pkl"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(USER_FOLDER, exist_ok=True)

# Load PCA model once
@st.cache_data
def load_pca_model(filename=MODEL_PATH):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model["mean_face"], model["eigenfaces"], model["weights"]

mean_face, eigenfaces, weights = load_pca_model()

def detect_and_crop_face(image_np, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image_np, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return image_np[y:y + h, x:x + w]

def process_image(image_file):
    image = Image.open(image_file).convert("L")
    image_np = np.array(image)
    cropped = detect_and_crop_face(image_np)
    if cropped is None:
        return None
    resized = Image.fromarray(cropped).resize((64, 64))
    return np.array(resized)

def save_user_face(username, face_image_np):
    user_dir = os.path.join(USER_FOLDER, username)
    os.makedirs(user_dir, exist_ok=True)
    save_path = os.path.join(user_dir, 'face.jpg')
    Image.fromarray(face_image_np).save(save_path)

def get_stored_users():
    users = {}
    for username in os.listdir(USER_FOLDER):
        user_dir = os.path.join(USER_FOLDER, username)
        face_path = os.path.join(user_dir, 'face.jpg')
        if os.path.isfile(face_path):
            users[username] = face_path
    return users

def load_user_face(face_path):
    img = Image.open(face_path).convert("L").resize((64, 64))
    return np.array(img)

def compare_faces(test_image, mean_face, eigenfaces, weights):
    test_vec = test_image.flatten().reshape(-1, 1)
    centered = test_vec - mean_face
    test_weights = eigenfaces @ centered
    sims = cosine_similarity(weights.T, test_weights.T).flatten()
    max_sim = np.max(sims)
    label_idx = np.argmax(sims) if max_sim >= 0.8 else None
    return label_idx, max_sim

# --- Streamlit UI ---
st.title("Face Recognition System")

action = st.radio("Choose Action", ["Sign Up", "Sign In"], horizontal=True)

if action == "Sign Up":
    st.header("Register New User")
    username = st.text_input("Enter your username")
    image_file = st.camera_input("Capture your face")

    if st.button("Register"):
        if not username:
            st.warning("Please enter a username.")
        elif image_file is None:
            st.warning("Please capture a face image.")
        else:
            face_img = process_image(image_file)
            if face_img is None:
                st.error("No face detected in the image. Try again.")
            else:
                save_user_face(username, face_img)
                st.success(f"User '{username}' registered successfully!")

elif action == "Sign In":
    st.header("User Login")
    username = st.text_input("Enter your username")
    image_file = st.camera_input("Capture your face")

    if st.button("Login"):
        if not username:
            st.warning("Please enter your username.")
        elif image_file is None:
            st.warning("Please capture your face image.")
        else:
            face_img = process_image(image_file)
            if face_img is None:
                st.error("No face detected. Please try again.")
            else:
                stored_users = get_stored_users()
                if username not in stored_users:
                    st.error(f"User '{username}' not found. Please sign up first.")
                else:
                    label_idx, similarity = compare_faces(face_img, mean_face, eigenfaces, weights)
                    user_list = list(stored_users.keys())
                    if label_idx is not None and user_list[label_idx] == username:
                        st.success(f"Welcome back, {username}! Face recognized with similarity {similarity:.2f}")
                    else:
                        st.error("Face not recognized or does not match the username.")
