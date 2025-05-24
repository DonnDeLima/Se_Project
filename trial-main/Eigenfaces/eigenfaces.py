import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import pickle

# Function to detect and crop face
def detect_and_crop_face(image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise IOError("Failed to load Haar cascade for face detection.")

    faces = face_cascade.detectMultiScale(
        image,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        return None  # No face detected

    x, y, w, h = faces[0]
    return image[y:y + h, x:x + w]

# Function to load images with face detection
def load_images_with_labels(folder, img_size):
    images = []
    labels = []
    label_map = {}
    current_label = 0

    for subdir in sorted(os.listdir(folder)):
        subdir_path = os.path.join(folder, subdir)
        if not os.path.isdir(subdir_path):
            continue

        label_map[current_label] = subdir
        for filename in sorted(os.listdir(subdir_path)):
            file_path = os.path.join(subdir_path, filename)
            try:
                img = Image.open(file_path).convert("L")  # Convert to grayscale
                img = np.array(img)
                cropped_face = detect_and_crop_face(img)
                if cropped_face is None:
                    print(f"No face detected in {filename}. Skipping.")
                    continue
                resized_face = Image.fromarray(cropped_face).resize((img_size, img_size))
                images.append(np.array(resized_face).flatten())  # Flatten for LDA
                labels.append(current_label)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
        current_label += 1
    return np.array(images), np.array(labels), label_map

# Fisherface Training
def train_fisherface_model(training_images, training_labels, num_components=None):
    lda = LDA(n_components=num_components)
    lda.fit(training_images, training_labels)
    transformed_data = lda.transform(training_images)
    return lda, transformed_data

# Save model to pickle
def save_model(lda, transformed_data, label_map, filename="fisherface_model.pkl"):
    model = {"lda": lda, "transformed_data": transformed_data, "label_map": label_map}
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}.")

# Load model from pickle
def load_model(filename="fisherface_model.pkl"):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {filename}.")
    return model["lda"], model["transformed_data"], model["label_map"]

# Fisherface Testing
def test_fisherface_model(test_image, lda, transformed_data, label_map, tolerance=0.8):
    transformed_test_image = lda.transform([test_image])
    similarities = cosine_similarity(transformed_data, transformed_test_image).flatten()
    max_similarity = np.max(similarities)
    recognized_label = np.argmax(similarities) if max_similarity >= tolerance else None
    return recognized_label, max_similarity, similarities

# Main Function
if __name__ == "__main__":
    img_size = 64  # Resize all images to 64x64
    training_folder = r"C:\Users\Joshua Ean\Desktop\Eigenfaces\dataset\ean"

    # Load training images and labels
    training_images, training_labels, label_map = load_images_with_labels(training_folder, img_size)

    if len(set(training_labels)) < 2:
        print("Error: At least two classes are required to train Fisherfaces.")
        exit()

    # Train the Fisherface model
    lda, transformed_data = train_fisherface_model(training_images, training_labels)

    # Save the model
    save_model(lda, transformed_data, label_map, filename="fisherface_model.pkl")

    # Load the model for testing
    lda, transformed_data, label_map = load_model(filename="fisherface_model.pkl")

    # Test images
    test_image_paths = [os.path.join(training_folder, f"{i + 1}.jpg") for i in range(5)]
    test_images = []
    for path in test_image_paths:
        try:
            img = Image.open(path).convert("L")
            img = np.array(img)
            cropped_face = detect_and_crop_face(img)
            if cropped_face is not None:
                resized_face = Image.fromarray(cropped_face).resize((img_size, img_size))
                test_images.append(np.array(resized_face).flatten())
            else:
                print(f"No face detected in test image: {path}")
        except Exception as e:
            print(f"Error loading test image {path}: {e}")

    # Test the model
    tolerance = 0.8  # Cosine similarity threshold
    for i, test_image in enumerate(test_images):
        recognized_label, similarity, all_similarities = test_fisherface_model(
            test_image, lda, transformed_data, label_map, tolerance
        )

        if recognized_label is None:
            print(
                f"Test Image {i + 1}: No match found. Max Similarity: {similarity:.4f}. "
                "Possible causes: face not in training set or low confidence."
            )
        else:
            print(
                f"Test Image {i + 1}: Recognized Label: {label_map[recognized_label]}, "
                f"Similarity: {similarity:.4f}"
            )
