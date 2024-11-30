This Python program simulates red-green color blindness (Deuteranopia) and provides a way to encode and decode images using Self-Organizing Maps (SOMs). It allows users to visually understand how individuals with color blindness perceive images and explore reconstruction techniques for color-blind-transformed images.
Features

    Simulate Deuteranopia:
        Applies a transformation matrix to simulate how images appear to individuals with red-green color blindness.

    Self-Organizing Map (SOM) Encoding:
        Encodes transformed images into a reduced representation using SOM, a type of unsupervised neural network.

    Decoding:
        Attempts to reconstruct original images from the encoded representation.

    Preprocessing:
        Resizes and normalizes images for efficient processing.

    Training:
        Trains SOM on both original and simulated images for better representation.

    Visualization:
        Displays the original, color-blindness-simulated, and SOM-processed images side-by-side for comparison.

Prerequisites

Before running the program, ensure you have the following installed:

    Python 3.7 or higher
    Required Python libraries: pip install numpy opencv-python matplotlib minisom
    
Usage
Step 1: Prepare Images

    Store your images locally and either:
        Update the PREDEFINED_IMAGE_PATHS list in the code with the file paths.
        Provide image paths during runtime.

Step 2: Run the Program

    Save the script as color_blindness_simulation.py.
    Execute the script: python finalSOM.py
    
Step 3: Choose an Option

    Option 1: Encode
    Simulates how the image appears to a color-blind individual.
    Option 2: Decode
    Attempts to reconstruct the original image from its encoded (simulated) version.
    Option 3: Use Predefined Image Paths
    Processes multiple images at once based on the predefined paths.

Step 4: Inspect Results

    View the images displayed by the program to compare transformations and reconstructions.

How It Works

    Simulating Color Blindness:
        A transformation matrix simulates Deuteranopia by altering RGB channels.

    Training SOM:
        Combines data from the original and color-blind images to train a Self-Organizing Map.

    Encoding and Decoding:
        Encodes images into the SOM's quantized space.
        Decodes images by mapping back from the SOM's space.

    Visualization:
        Uses Matplotlib to plot and compare the original, simulated, and processed images.

Customization

    Adjusting SOM Size:
        Modify the som_size parameter in train_on_multiple_images() to control SOM grid resolution.

    Training Iterations:
        Adjust the number of iterations (5000 by default) to improve SOM accuracy.

    Image Resizing:
        Change the target_size in the resize_image() function to match your desired output dimensions.

Known Limitations

    Decoding may not perfectly reconstruct original images due to information loss during encoding.
    Large images can slow processing; resizing to 256x256 pixels is recommended.
    Image paths must be valid and accessible.
