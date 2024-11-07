import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# Function to extract features
def extract_features(image):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    #  color features
    mean_hue = np.mean(hsv_image[:, :, 0])
    mean_saturation = np.mean(hsv_image[:, :, 1])
    mean_value = np.mean(hsv_image[:, :, 2])
    std_hue = np.std(hsv_image[:, :, 0])
    std_saturation = np.std(hsv_image[:, :, 1])
    std_value = np.std(hsv_image[:, :, 2])

    #  texture features (using LBP)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = cv2.calcHist([gray_image], [0], None, [256], [0, 256])  # Using histogram as a simple feature
    contrast = np.mean(lbp)
    energy = np.sum(lbp ** 2)

    #  shape features
    gray_image_bin = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(gray_image_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = sum(cv2.contourArea(c) for c in contours)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    aspect_ratio = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0

    return np.array([
        mean_hue, mean_saturation, mean_value,
        std_hue, std_saturation, std_value,
        contrast, energy,
        area, perimeter, aspect_ratio
    ])

# Loading images and labels
def load_data(dataset_path):
    images = []
    labels = []
    features = []

    # Loading good quality fruits
    good_path = os.path.join(dataset_path, 'Good Quality_Fruits')
    for category in os.listdir(good_path):
        category_path = os.path.join(good_path, category)
        if os.path.isdir(category_path):
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(img_path)
                    img_resized = cv2.resize(img, (150, 150))
                    images.append(img_resized)
                    labels.append(1)  # Good quality
                    features.append(extract_features(img_resized))

    # Loading bad quality fruits
    bad_path = os.path.join(dataset_path, 'Bad Quality_Fruits')
    for category in os.listdir(bad_path):
        category_path = os.path.join(bad_path, category)
        if os.path.isdir(category_path):
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(img_path)
                    img_resized = cv2.resize(img, (150, 150))
                    images.append(img_resized)
                    labels.append(0)  # Bad quality
                    features.append(extract_features(img_resized))

    print(f"Total images loaded: {len(images)}")
    return np.array(images), np.array(labels), np.array(features)

# Set the dataset path
dataset_path = "C:/Users/ASHISH/trial/dataset"

# Load the data
X_images, y, extracted_features = load_data(dataset_path)

# Check if any images were loaded
if len(X_images) == 0:
    print("No images loaded. Please check your dataset path and structure.")
else:
    # Normalizing the pixel values for images
    X_images = X_images.astype('float32') / 255.0

    # Train-test split
    X_train, X_test, y_train, y_test, features_train, features_test = train_test_split(
        X_images, y, extracted_features, test_size=0.2, random_state=42
    )

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

    # Compiling the model and Fit the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

    # Save the model with a valid extension
    model_save_path = "C:/Users/ASHISH/trial/my_fruit_quality_model.h5"
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")


    predictions = model.predict(X_test)
    percentage_quality = predictions * 100  # Converting to percentage
    print("Predicted percentage quality:", percentage_quality)
