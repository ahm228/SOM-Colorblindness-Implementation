import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Predefined paths for initial batch of images
PREDEFINED_IMAGE_PATHS = [
     r'C:\Users\kmang\OneDrive\Desktop\Color Blind\1.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\2.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\3.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\4.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\5.jpg',
]

# Function to simulate red-green color blindness (Deuteranopia)
def simulate_color_blindness(image):
    matrix = np.array([[0.56667, 0.43333, 0],
                       [0.55833, 0.44167, 0],
                       [0, 0.24167, 0.75833]])
    return np.dot(image, matrix.T)

# Function to load and preprocess a single image
def load_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image / 255.0  # Normalize to [0, 1]

# Build the autoencoder model
def build_autoencoder(input_shape):
    encoder = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
    ])

    decoder = models.Sequential([
        layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same'),
    ])

    autoencoder = models.Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Function to display the results
def display_results(original, transformed, reconstructed):
    plt.figure(figsize=(10, 4))  # Reduced figure size for a more compact display

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(original)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Color Blind")
    plt.imshow(transformed)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Reconstructed")
    plt.imshow(reconstructed)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Build the autoencoder
input_shape = (256, 256, 3)
autoencoder = build_autoencoder(input_shape)

# Continuous Training Loop
print("Starting continuous image loading and training...")
for image_path in PREDEFINED_IMAGE_PATHS:
    print(f"Processing image: {image_path}")
    
    # Load and process image
    original_image = load_image(image_path)
    color_blind_image = simulate_color_blindness(original_image)
    
    # Expand dimensions for batch training
    original_batch = np.expand_dims(original_image, axis=0)
    color_blind_batch = np.expand_dims(color_blind_image, axis=0)

    # Train the autoencoder on this single image batch
    autoencoder.fit(color_blind_batch, original_batch, epochs=300, batch_size=1, verbose=1)
    
    # Display the reconstructed image for visual feedback
    reconstructed_image = autoencoder.predict(color_blind_batch)[0]
    display_results(original_image, color_blind_image, reconstructed_image)

print("Training completed.")
