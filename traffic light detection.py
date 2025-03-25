import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('C:/Users/shana/Desktop/Traffic signs and light detection system/traffic_light_model.keras')
# Define the class labels
class_labels = [ 'NO SIGN DETECTED', 'TLS-R', 'TLS-G', 'TLS-Y']

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input size
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Start capturing video from the camera using OpenCV
video_capture = cv2.VideoCapture(0)  # Use 0 for the default camera

if not video_capture.isOpened():
    print("Error: Could not open video.")
    exit()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip the frame horizontally for mirror view
        frame = cv2.flip(frame, 1)  # 1 means flipping around y-axis

        # Convert the frame to a PIL image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Preprocess the image
        processed_image = preprocess_image(pil_image)

        # Make predictions (Split the outputs: class and bounding box)
        class_pred, bbox_pred = model.predict(processed_image)

        # Get the predicted class
        predicted_class = np.argmax(class_pred, axis=1)[0]
        predicted_label = class_labels[predicted_class]

        # Display the label on the image
        cv2.putText(frame, f"Detected: {predicted_label}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame with the label
        cv2.imshow('Traffic Light Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    video_capture.release()
    cv2.destroyAllWindows()
