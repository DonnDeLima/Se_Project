import cv2
import numpy as np
import os

def capture_images(num_images=5, img_size=(200, 200), save_path="dataset"):
    """
    Captures images for multiple classes using the webcam and saves them in labeled folders.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    while True:
        person_name = input("Enter the person's name (or type 'done' to finish): ").strip()
        if person_name.lower() == 'done':
            print("Finished capturing images.")
            break

        person_path = os.path.join(save_path, person_name)
        if not os.path.exists(person_path):
            os.makedirs(person_path)

        cap = cv2.VideoCapture(0)
        print(f"Capturing {num_images} images for {person_name}. Press 'q' to quit early.")

        count = 0
        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, img_size)

            cv2.imshow("Capture Image", frame)

            # Save the image
            file_path = os.path.join(person_path, f"{count + 1}.jpg")
            cv2.imwrite(file_path, resized)
            print(f"Saved: {file_path}")

            count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting early...")
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"Image capture complete for {person_name}.")

def read_images(data_path, img_size=(200, 200)):
    """
    Reads images from the dataset folder and returns them along with labels.
    """
    images = []
    labels = []
    label_map = {}
    current_label = 0

    for root, dirs, files in os.walk(data_path):
        for dir_name in dirs:
            label_map[current_label] = dir_name
            subject_path = os.path.join(root, dir_name)
            for file_name in os.listdir(subject_path):
                file_path = os.path.join(subject_path, file_name)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, img_size)
                    images.append(img_resized)
                    labels.append(current_label)
            current_label += 1

    return np.asarray(images), np.asarray(labels), label_map

def test_model(fisherface, label_map, img_size=(200, 200)):
    """
    Captures a test image and predicts its label using the trained Fisherface model.
    """
    cap = cv2.VideoCapture(0)
    print("Align your face and press 's' to capture a test image.")

    test_img = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, img_size)

        cv2.imshow("Test Image Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            test_img = resized
            print("Test image captured.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if test_img is not None:
        label, confidence = fisherface.predict(test_img)
        print(f"Predicted Label: {label_map[label]}, Confidence: {confidence}")
    else:
        print("No test image captured.")

def main():
    data_path = "dataset"
    img_size = (200, 200)

    # Step 1: Capture training images
    print("Step 1: Capture Training Images")
    capture_images(num_images=5, img_size=img_size, save_path=data_path)

    # Step 2: Read and train the Fisherface model
    print("\nStep 2: Train the Fisherface Model")
    images, labels, label_map = read_images(data_path, img_size)

    # Check if there are at least two classes
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print("Error: At least two classes are needed to train the model. Capture images for another person.")
        return

    fisherface = cv2.face.FisherFaceRecognizer_create()
    fisherface.train(images, labels)
    print("Fisherface model trained successfully.")

    # Step 3: Test the model
    print("\nStep 3: Test the Model with a Live Capture")
    test_model(fisherface, label_map, img_size)

if __name__ == "__main__":
    main()
