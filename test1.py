import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
import time
from keras.api.models import load_model
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import pickle

# Load the pre-trained cascade classifier for detecting faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained face recognition model
model = load_model(r'C:\Users\user\Desktop\AI_test\face_recognition_model.keras')

# Load class indices (assuming you saved this during training)
with open('class_indices.pkl', 'rb') as f:
    class_indices = pickle.load(f)

# Invert class indices dictionary to map indices to class names
class_names = {v: k for k, v in class_indices.items()}

# Path to save the images
save_path = "C:\\Users\\user\\Desktop\\faces"

# Ensure the save path exists
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Function to assign a random point to a face
def assign_random_point(x, y, w, h):
    random_x = np.random.randint(x, x + w)
    random_y = np.random.randint(y, y + h)
    return (random_x, random_y)

# Function to preprocess the face for the model
def preprocess_face(face):
    face = cv2.resize(face, (64, 64))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    return face

# Function to capture photos
def photo_capture_mode(cap):
    face_points = {}
    face_names = {}
    last_capture_time = time.time()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Store new face points
        new_face_points = {}

        # Draw rectangles around the faces and assign points
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Generate a unique key for the face based on position
            face_key = (x, y, w, h)

            if face_key not in face_points:
                # Assign a random point if this face is new
                face_points[face_key] = assign_random_point(x, y, w, h)
                print("New person detected! Please enter a name or type 'so' to skip:")

                # Display the frame with the new face
                cv2.imshow('Face Detection with Random Points', frame)
                cv2.waitKey(1)  # Small wait to ensure the frame is displayed

                # Get user input for the new person's name
                person_name = input("Enter name for the new person: ")

                if person_name.lower() != 'so':
                    # Save the photo with the person's name
                    cv2.imwrite(os.path.join(save_path, f"{person_name}.jpg"), frame)

                    # Store the name in the face_names dictionary
                    face_names[face_key] = person_name
                else:
                    # Automatically use the model to recognize the face
                    face_roi = frame[y:y+h, x:x+w]
                    preprocessed_face = preprocess_face(face_roi)
                    prediction = model.predict(preprocessed_face)
                    predicted_class = np.argmax(prediction)

                    # Handle KeyError for the predicted class
                    predicted_name = class_names.get(predicted_class, "Unknown")
                    face_names[face_key] = predicted_name

            # Draw the random point on the face
            random_point = face_points[face_key]
            cv2.circle(frame, random_point, 5, (0, 255, 0), -1)

            # Update new face points dictionary
            new_face_points[face_key] = face_points[face_key]

            # Preprocess the face for prediction if not done already
            if face_key in face_names and face_names[face_key] != "Unknown":
                face_roi = frame[y:y+h, x:x+w]
                preprocessed_face = preprocess_face(face_roi)
                prediction = model.predict(preprocessed_face)
                predicted_class = np.argmax(prediction)

                # Handle KeyError for the predicted class
                predicted_name = class_names.get(predicted_class, "Unknown")

                # Update the name if the AI detects a match
                if predicted_name != "Unknown":
                    face_names[face_key] = predicted_name

            # Display the name on the frame
            cv2.putText(frame, face_names[face_key], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Update the face points with the new detected faces
        face_points = new_face_points

        # Display the resulting frame
        cv2.imshow('Face Detection with Random Points', frame)

        # Capture and save an image every 5 seconds
        current_time = time.time()
        if current_time - last_capture_time >= 5:
            last_capture_time = current_time
            for face_key in new_face_points:
                if face_key not in face_names:
                    continue
                person_name = face_names[face_key]
                cv2.imwrite(os.path.join(save_path, f"{person_name}_{int(current_time)}.jpg"), frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Function to recognize faces
def camera_mode(cap):
    face_names = {}
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the faces and display names
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Preprocess the face for prediction
            face_roi = frame[y:y+h, x:x+w]
            preprocessed_face = preprocess_face(face_roi)
            prediction = model.predict(preprocessed_face)
            predicted_class = np.argmax(prediction)

            # Handle KeyError for the predicted class
            predicted_name = class_names.get(predicted_class, "Unknown")

            # Display the name on the frame
            cv2.putText(frame, predicted_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Display the resulting frame
        cv2.imshow('Face Recognition', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Main function to choose mode
def main():
    cap = cv2.VideoCapture(0)
    mode = input("Enter mode ('capture' for photo capture mode, 'camera' for camera mode): ").strip().lower()

    if mode == 'capture':
        photo_capture_mode(cap)
    elif mode == 'camera':
        camera_mode(cap)
    else:
        print("Invalid mode selected. Please enter 'capture' or 'camera'.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
