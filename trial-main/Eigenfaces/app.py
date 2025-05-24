from flask import Flask, render_template, request, flash, session
import os
import cv2
import numpy as np
import pickle
from PIL import Image
import io
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flashing messages

# Folder paths
UPLOAD_FOLDER = 'static/uploads'
USER_FOLDER = 'data/users'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(USER_FOLDER):
    os.makedirs(USER_FOLDER)

# Load the PCA model
def load_model(filename="Eigenfaces/model/fisherface_model.pkl"):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model["mean_face"], model["eigenfaces"], model["weights"]

# Function to detect and crop face from an image
def detect_and_crop_face(image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise IOError("Failed to load Haar cascade for face detection.")
    faces = face_cascade.detectMultiScale(image, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size, flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return image[y:y + h, x:x + w]

# Function to process uploaded image
def process_uploaded_image(image_data):
    image = Image.open(io.BytesIO(image_data)).convert("L")
    image = np.array(image)
    cropped_face = detect_and_crop_face(image)
    if cropped_face is None:
        return None
    img_size = 64
    resized_face = Image.fromarray(cropped_face).resize((img_size, img_size))
    return np.array(resized_face)

# Save user's face data (image and name)
def save_user(name, face_image):
    user_dir = os.path.join(USER_FOLDER, name)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    file_path = os.path.join(user_dir, 'face.jpg')
    Image.fromarray(face_image).save(file_path)

# Function to retrieve all stored users
def get_stored_users():
    users = {}
    for user_name in os.listdir(USER_FOLDER):
        user_dir = os.path.join(USER_FOLDER, user_name)
        if os.path.isdir(user_dir):
            users[user_name] = user_dir
    return users

# Function to compare the captured face with stored user faces
def compare_faces(test_image, mean_face, eigenfaces, weights):
    test_vector = test_image.flatten().reshape(-1, 1)
    centered_test_vector = test_vector - mean_face
    test_weights = eigenfaces @ centered_test_vector
    similarities = cosine_similarity(weights.T, test_weights.T).flatten()
    max_similarity = np.max(similarities)
    recognized_label = np.argmax(similarities) if max_similarity >= 0.8 else None
    return recognized_label, max_similarity

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/signin', methods=['POST'])
def signin():
    user_name = request.form['username']
    
    # Load the PCA model
    mean_face, eigenfaces, weights = load_model("Eigenfaces\model\pca_model.pkl")

    # Capture the live face image
    return render_template('signin.html', user_name=user_name)

@app.route('/signup', methods=['POST'])
def signup():
    user_name = request.form['username']
    return render_template('signup.html', user_name=user_name)

@app.route('/capture_signin', methods=['POST'])
def capture_signin():
    user_name = request.form['username']
    
    # Get the live webcam image from the form
    image_data = request.files['file'].read()
    processed_image = process_uploaded_image(image_data)

    if processed_image is None:
        flash("No face detected. Please try again.", 'error')
        return render_template('dashboard.html', user_name=user_name)
    
    # Load the PCA model
    mean_face, eigenfaces, weights = load_model("Eigenfaces\model\pca_model.pkl")
    
    # Compare captured face with stored faces
    stored_users = get_stored_users()
    recognized_label, similarity = compare_faces(processed_image, mean_face, eigenfaces, weights)
    
    # if recognized_label is None or list(stored_users.keys()) != user_name:     #or list(stored_users.keys()) #[recognized_label]
    #     # If user is not recognized, redirect back with error message
    #     flash(f"User {user_name} is not registered. Please register first!", 'error')
    #     return render_template('dashboard.html', user_name=user_name)
    if recognized_label is None or user_name not in stored_users:  
        # If user is not recognized or is not in the stored users
        flash(f"User {user_name} is not registered. Please register first!", 'error')
        return render_template('dashboard.html', user_name=user_name)
    
    return render_template('vault.html', user_name=user_name)

@app.route('/capture_signup', methods=['POST'])
def capture_signup():
    user_name = request.form['username']
    
    # Get the live webcam image from the form
    image_data = request.files['file'].read()
    processed_image = process_uploaded_image(image_data)

    if processed_image is None:
        flash("No face detected. Please try again.", 'error')
        return render_template('dashboard.html', user_name=user_name)

    # Save the user's face image and name to the database
    save_user(user_name, processed_image)
    
    # Flash a success message
    flash(f"User {user_name} registered successfully!", 'success')

    # Redirect to the dashboard page
    return render_template('dashboard.html', user_name=user_name)

# Logout route
@app.route('/logout', methods=['POST'])
def logout():
    # Clear the user's session
    session.clear()
    flash("You have been successfully logged out.", 'success')
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)
