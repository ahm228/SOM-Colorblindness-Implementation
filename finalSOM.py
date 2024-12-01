import numpy as np
import cv2
import matplotlib.pyplot as plt
from minisom import MiniSom
import os


# Predefined list of image paths
PREDEFINED_IMAGE_PATHS = [
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\1.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\2.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\3.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\4.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\5.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\6.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\7.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\8.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\9.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\10.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\11.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\12.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\13.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\14.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\15.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\16.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\17.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\18.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\19.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\20.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\21.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\22.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\23.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\24.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\25.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\26.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\27.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\28.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\29.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\30.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\31.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\32.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\33.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\34.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\35.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\36.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\37.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\38.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\39.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\40.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\41.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\42.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\43.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\44.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\45.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\46.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\47.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\48.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\49.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\50.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\51.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\52.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\53.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\54.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\55.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\56.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\57.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\58.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\59.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\60.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\61.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\62.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\63.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\64.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\65.jpg',
    r'C:\Users\kmang\OneDrive\Desktop\Color Blind\66.jpg',
]


# Function to simulate red-green color blindness (Deuteranopia)
def simulate_color_blindness(image):
    matrix = np.array([[0.56667, 0.43333, 0],
                       [0.55833, 0.44167, 0],
                       [0, 0.24167, 0.75833]])
    return np.dot(image, matrix.T)

# Function to normalize pixel values to 0-1
def normalize_image(image):
    return np.clip(image.astype('float32') / 255, 0, 1)

# Function to denormalize the values back to 0-255
def denormalize_image(image):
    return np.clip(image * 255, 0, 255).astype('uint8')

# Function resizes the images to be 256x256 pixels
def resize_image(image, target_size=(256, 256)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

# Function to train the SOM with multiple images
def train_on_multiple_images(image_paths, som_size=16):
    all_training_data = []
    original_to_deuteranopia = []

    for image_path in image_paths:
        # Process the original input image
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image = resize_image(original_image)
        original_image = normalize_image(original_image)

        # Transform original to Deuteranopia image simulation
        deuteranopia_image = simulate_color_blindness(original_image)

        # Flatten and concatenate for training
        reshaped_original = original_image.reshape(-1, 3)
        reshaped_deuteranopia = deuteranopia_image.reshape(-1, 3)
        concatenated_data = np.concatenate((reshaped_original, reshaped_deuteranopia))

        all_training_data.append(concatenated_data)
        original_to_deuteranopia.append((reshaped_original, reshaped_deuteranopia))

    # Combine all data
    all_training_data = np.vstack(all_training_data)

    # Train SOM
    som = MiniSom(som_size, som_size, 3, sigma=1.0, learning_rate=0.5)
    som.random_weights_init(all_training_data)
    som.train_random(all_training_data, 5000)

    # Store mapping for decoding
    return som, original_to_deuteranopia

# Encoding changes original to the deuteranopia image
def encode_with_som(image, som, batch_size=1024):
    reshaped_image = normalize_image(image).reshape(-1, 3)
    encoded_image = []

    for i in range(0, len(reshaped_image), batch_size):
        batch = reshaped_image[i:i + batch_size]
        encoded_batch = som.quantization(batch)
        encoded_image.extend(encoded_batch)

    return np.array(encoded_image).reshape(image.shape)

def decode_with_som(encoded_image, som, original_to_deuteranopia, som_size=16):
    reshaped_encoded = normalize_image(encoded_image).reshape(-1, 3)
    som_weights = som.get_weights()

    # Create mapping from BMU indices to original pixels
    weight_to_original = {}
    for original, deuteranopia in original_to_deuteranopia:
        for orig_pixel, deut_pixel in zip(original, deuteranopia):
            bmu = som.winner(deut_pixel)
            weight_to_original[bmu] = orig_pixel

    # Default mapping for missing BMUs
    for x in range(som_size):
        for y in range(som_size):
            bmu = (x, y)
            if bmu not in weight_to_original:
                weight_to_original[bmu] = som_weights[x, y]  # Default to SOM weights

    # Decode the image
    decoded_image = []
    for pixel in reshaped_encoded:
        bmu = som.winner(pixel)
        decoded_image.append(weight_to_original.get(bmu, pixel))  # Default to input pixel if no match

    # Denormalize and reshape to original dimensions
    return denormalize_image(np.array(decoded_image).reshape(encoded_image.shape))

# Function to display the output of what the SOM does (commented out values just change whats outputted)
def display_and_save_images(original, transformed, output, mode):
    plt.figure(figsize=(10, 5)) #(15,5)

    plt.subplot(1, 2, 1) #(1,3,1)
    plt.title("Original Image")
    plt.imshow(original)

    plt.subplot(1, 2, 2) #(1,3,2)
    plt.title(f"{mode} Image")
    plt.imshow(transformed)

    # plt.subplot(1, 3, 3)
    # plt.title("Output Image")
    # plt.imshow(output)

    plt.tight_layout()
    plt.show()

# Function to process the image based on encode or decode
def process_image(image_path, mode, som, original_to_deuteranopia=None):
    # Load and preprocess the original image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    resized_image = resize_image(original_image)

    if mode == 'encode':
        # Simulate color blindness and encode
        color_blind_image = simulate_color_blindness(normalize_image(resized_image))
        encoded_image = encode_with_som(color_blind_image, som)

        display_and_save_images(
            resized_image,
            denormalize_image(color_blind_image),
            denormalize_image(encoded_image),
            "Encoded"
        )

    elif mode == 'decode':
        if original_to_deuteranopia is None:
            raise ValueError("Original-to-Deuteranopia mapping is required for decoding.")

        decoded_image = decode_with_som(normalize_image(resized_image), som, original_to_deuteranopia)

        display_and_save_images(
            resized_image,
            denormalize_image(resized_image),
            denormalize_image(decoded_image),
            "Decoded"
        )

# Command-line interface
def main():
    print("Welcome to the Color Blindness Simulation Program!\n")
    print("1. Encode (Original -> Color Blindness Simulation)")
    print("2. Decode (Color Blindness Simulation -> Original)")
    print("3. Use Predefined Image Paths\n")

    choice = input("Select an option (1, 2, or 3): ")

    if choice == '1':
        mode = 'encode'
        image_paths = input("Enter the paths of image files separated by commas: ").split(',')
        image_paths = [path.strip() for path in image_paths]

    elif choice == '2':
        mode = 'decode'
        image_paths = input("Enter the paths of image files separated by commas: ").split(',')
        image_paths = [path.strip() for path in image_paths]

    elif choice == '3':
        mode = 'encode'  # or 'decode' to decode multiple at once
        image_paths = PREDEFINED_IMAGE_PATHS

    else:
        print("Invalid option selected. Exiting.")
        return

    if all(os.path.exists(path) for path in image_paths):
        print("Training SOM... Please wait.")
        som, original_to_deuteranopia = train_on_multiple_images(image_paths)

        for image_path in image_paths:
            print(f"\nProcessing {image_path}...")
            process_image(image_path, mode, som, original_to_deuteranopia if mode == 'decode' else None)
    else:
        print("One or more files not found. Please try again.")

if __name__ == '__main__':
    main()
