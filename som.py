import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from PIL import Image

# Color dataset with descriptive labels
color_data = [
    ([255, 0, 0], "Red"),
    ([0, 255, 0], "Green"),
    ([0, 0, 255], "Blue"),
    ([255, 255, 0], "Yellow"),
    ([0, 255, 255], "Cyan"),
    ([255, 0, 255], "Magenta"),
    #([128, 128, 128], "Gray"),
    ([0, 0, 0], "Black"),
    ([255, 255, 255], "White"),
]

# Prepare the dataset
colors = np.array([item[0] for item in color_data]) / 255  # Normalize RGB values
labels = [item[1] for item in color_data]

# Initialize SOM
som = MiniSom(x=3, y=3, input_len=3, sigma=1.0, learning_rate=0.5, random_seed=42)
som.random_weights_init(colors)
som.train_random(colors, 100)

# Function to find the closest matching color and label
def find_closest_color(som, input_color):
    input_color = np.array(input_color) / 255
    winner = som.winner(input_color)
    weights = som.get_weights()
    closest_color = weights[winner]
    closest_color = (closest_color * 255).astype(int)
    distances = np.linalg.norm(colors - closest_color / 255, axis=1)
    closest_label = labels[np.argmin(distances)]
    return closest_label, closest_color

# Function to process an image and recognize colors
def process_image(image_path, som):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((256, 256))  # Resize for faster processing
    image_data = np.array(image)

    recognized_colors = []
    for row in image_data:
        for pixel in row:
            pixel = pixel[:3]  # Ensure pixel is RGB (ignore alpha if present)
            label, matched_color = find_closest_color(som, pixel)
            recognized_colors.append((tuple(pixel), label))

    # Visualize the result
    output_image = np.zeros_like(image_data)
    for i, row in enumerate(image_data):
        for j, pixel in enumerate(row):
            pixel = pixel[:3]  # Ensure pixel is RGB
            _, matched_color = find_closest_color(som, pixel)
            output_image[i, j] = matched_color

    return Image.fromarray(output_image.astype(np.uint8)), recognized_colors

# Prompt user for the image path
image_path = input("Please enter the path to your image: ")
try:
    output_image, recognized_colors = process_image(image_path, som)

    # Save and display the result
    output_image.save("output_image.jpg")
    output_image.show()

    # Print recognized colors (optional)
    for original, label in recognized_colors[:10]:  # Display first 10 for brevity
        print(f"Original Color: {original}, Recognized Label: {label}")
except FileNotFoundError:
    print("The specified image file was not found. Please check the path and try again.")
except Exception as e:
    print(f"An error occurred: {e}")

# Visualization of SOM weights
plt.figure(figsize=(5, 5))
for i in range(3):
    for j in range(3):
        plt.subplot(3, 3, i * 3 + j + 1)
        plt.imshow([[som.get_weights()[i, j]]])
        plt.axis('off')
plt.suptitle("SOM Color Map")
plt.show()
