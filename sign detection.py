import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('C:/Users/shana/Desktop/Traffic signs and light detection system/New_traffic_sign_model.keras')

# Define the class labels
class_labels = [
    'Bus-stop', 'Children-Crossing-Ahead', 'Compulsory-Roundabout',
    'Cross-Roads-Ahead', 'Double-Bend-to-Left-Ahead',
    'Double-Bend-to-Right-Ahead', 'Falling-Rocks-Ahead', 'Left-Bend-Ahead',
    'Level-crossing-with-barriers-ahead',
    'Level-crossing-without-barriers-ahead', 'Narrow-Bridge-or-Culvert-Ahead',
    'No-entry', 'No-horns', 'No-left-turn', 'No-parking',
    'No-parking-and-standing', 'No-parking-on-even-numbered-days',
    'No-parking-on-odd-numbered-days', 'No-right-turn', 'No-u-turn',
    'Pass-onto-left', 'Pass-onto-right', 'Pedestrian-Crossing',
    'Pedestrian-crossing-Ahead', 'Proceed-straight', 'Right-Bend-Ahead',
    'Road-Bump-Ahead', 'Road-Closed-for-All-Vehicles', 'Roundabout-Ahead',
    'Speed-Limit-40-Kmph', 'Speed-Limit-50-Kmph', 'Speed-Limit-60-Kmph', 'Stop',
    'T-Junction-Ahead', 'Traffic-from-left-merges-ahead',
    'Traffic-from-right-merges-ahead', 'Turn-left', 'Turn-left-ahead',
    'Turn-right', 'Turn-right-ahead', 'hospital',
]

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to match model input size
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

        # Make predictions
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Check if the predicted class is valid
        if predicted_class < len(class_labels):
            predicted_label = class_labels[predicted_class]
        else:
            predicted_label = 'No sign detected'

        # Display the label on the image
        cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show the frame with the label
        cv2.imshow('Traffic Sign Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    video_capture.release()
    cv2.destroyAllWindows()