import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf

# Load your models
traffic_sign_model = tf.keras.models.load_model('C:/Users/shana/Desktop/Traffic signs and light detection system/New_traffic_sign_model.keras')
traffic_light_model = tf.keras.models.load_model('C:/Users/shana/Desktop/Traffic signs and light detection system/traffic_light_model.keras')

# Define the class labels for traffic signs and traffic lights
traffic_sign_labels = ['Bus-stop', 'Children-Crossing-Ahead', 'Compulsory-Roundabout',
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
    'Turn-right', 'Turn-right-ahead', 'hospital']  # Add all your actual class labels here

traffic_light_labels = ["No Light Detected", "Red", "Green", "Yellow"]

# Initialize the main window
app = tk.Tk()
app.title("Traffic Sign & Light Detection System")
app.geometry("1000x600")

# Global variables
camera = None
camera_label = None
is_camera_running = False

# Function to prepare image for prediction
def prepare_image(image):
    image_resized = cv2.resize(image, (224, 224))  # Resize to 128*128
    image_array = np.array(image_resized) / 255.0  # Normalize pixel values
    image_array = image_array.astype(np.float32)  # Ensure it's float32 for the model
    image_array = np.expand_dims(image_array, axis=0)  # Reshape to (1, 128, 128, 3)
    return image_array

# Open Camera Function
def open_camera():
    global camera, is_camera_running, camera_label

    if not is_camera_running:
        camera = cv2.VideoCapture(0)
        is_camera_running = True
        update_camera()

def update_camera():
    global camera, camera_label, is_camera_running

    if is_camera_running and camera.isOpened():
        ret, frame = camera.read()
        if ret:
            # Resize the frame for display
            frame_resized = cv2.resize(frame, (400, 300))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)

            # Update camera label
            if camera_label is None:
                camera_label = tk.Label(app)
                camera_label.place(x=10, y=50)
            camera_label.configure(image=img_tk)
            camera_label.image = img_tk

        # Continue updating the camera
        app.after(10, update_camera)

def stop_camera():
    global camera, is_camera_running

    if is_camera_running:
        camera.release()
        is_camera_running = False

def display_image(image):
    image_resized = image.resize((400, 300))
    image_tk = ImageTk.PhotoImage(image=image_resized)
    display_label.config(image=image_tk)
    display_label.image = image_tk

def predict_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    try:
        # Open the image using OpenCV
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_array = prepare_image(image)

        # Make predictions with both models
        traffic_sign_prediction = traffic_sign_model.predict(image_array)
        traffic_light_prediction = traffic_light_model.predict(image_array)

        # Get predicted labels
        sign_label = traffic_sign_labels[np.argmax(traffic_sign_prediction)]
        light_label = traffic_light_labels[np.argmax(traffic_light_prediction)]

        # Display prediction result
        prediction_label.config(text=f'Traffic Sign: {sign_label}\nTraffic Light: {light_label}')

        # Display the image in the interface
        display_image(Image.open(file_path))

    except Exception as e:
        messagebox.showerror("Error", f"Failed to process the image.\n{e}")

# Camera Control Buttons
btn_open_camera = tk.Button(app, text="Open Camera", command=open_camera)
btn_open_camera.place(x=10, y=10)

btn_stop_camera = tk.Button(app, text="Stop Camera", command=stop_camera)
btn_stop_camera.place(x=120, y=10)

# Image Prediction Button
btn_import_image = tk.Button(app, text="Import Image & Predict", command=predict_image)
btn_import_image.place(x=230, y=10)

# Result Label
prediction_label = tk.Label(app, text="", font=("Arial", 12))
prediction_label.place(x=450, y=50)

# Image Display Label
display_label = tk.Label(app)
display_label.place(x=10, y=150)

# Close the camera properly when closing the app
def on_closing():
    stop_camera()
    app.destroy()

app.protocol("WM_DELETE_WINDOW", on_closing)

# Run the application
app.mainloop()
