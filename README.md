Overview

This project implements an autoencoder neural network using TensorFlow and Keras to reconstruct images perceived under simulated red-green color blindness (Deuteranopia). The model is trained to transform a color-blind view of an image back into its original representation, providing an enhanced visualization for individuals with color blindness.
Features

    Color Blindness Simulation: Simulates red-green color blindness using a color transformation matrix.
    Autoencoder Architecture: A convolutional autoencoder is used for reconstructing images.
    Real-Time Feedback: Visualizes the original, color-blind simulated, and reconstructed images during training.
    Predefined Batch Training: Loads and processes images from a predefined set of paths for continuous training.

Prerequisites

    TensorFlow (tensorflow)
    NumPy (numpy)
    OpenCV (cv2)
    Matplotlib (matplotlib)

How to Use

    Setup Image Paths: Replace the PREDEFINED_IMAGE_PATHS list with paths to chosen images (I provided all images used with model testing)
    Run the Script: Execute the Python script to start the training process.
    Visual Feedback: After each training iteration, the script will display:
        The original image.
        The simulated color-blind image.
        The reconstructed image.
    To see other images listed if more than one, simply press the X on the output and the code will run through 300 epochs again for the next image
    Recommended: only testing 5 images at a time, unless you have the time to sit and wait on the progam to run

Key Functions
1. simulate_color_blindness(image)

Simulates red-green color blindness (Deuteranopia) using a transformation matrix.
2. load_image(image_path, target_size=(256, 256))

Loads and preprocesses an image:

    Converts to RGB.
    Resizes to the target dimensions.
    Normalizes pixel values to [0, 1].

3. build_autoencoder(input_shape)

Defines the autoencoder architecture:

    Encoder: Two convolutional layers with max-pooling.
    Decoder: Two deconvolutional layers with upsampling.

4. display_results(original, transformed, reconstructed)

Displays the original, color-blind simulated, and reconstructed images side-by-side using Matplotlib.
Model Training

The model is trained continuously on individual image batches:

    Input: Simulated color-blind images.
    Target: Original images.
    Loss: Mean squared error (MSE).

Each image batch undergoes 300 epochs of training, and reconstructed results are displayed after training.

Customization

    Adjust Input Shape: Modify input_shape to fit your dataset dimensions.
    Change Training Parameters: Update the number of epochs or batch size to suit your hardware.
    Add More Images: Extend the PREDEFINED_IMAGE_PATHS list with more image paths.

Future Work

    Extend the model to handle other types of color blindness (e.g., Protanopia, Tritanopia).
    Enhance model accuracy with larger datasets and more diverse training samples.
    Deploy the model as a web or mobile application for real-time image correction.
