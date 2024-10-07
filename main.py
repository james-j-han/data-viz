import os
import glob
from PIL import Image
from PIL import ImageTk
import time
import torch
from torchvision import models, transforms
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.offsetbox as offsetbox
import tkinter as tk
from tkinter import filedialog, Toplevel, Label, Button
from sklearn.metrics.pairwise import euclidean_distances

# Function to upload a single image
def upload_and_query_image(top_k=5):
    # Use Tkinter to open a file picker dialog
    # root = tk.Tk()
    # root.withdraw()  # Hide the root window
    img_path = filedialog.askopenfilename(title="Select an image to query")
    if not img_path:
        print("No image selected")
        return None

    print(f'Uploaded image: {img_path}')
    
    # Extract features from the uploaded image
    query_img = load_image(img_path)
    with torch.no_grad():
        query_feature = model(query_img).squeeze().numpy()

    # Calculate distances and find top-k closest images
    distances = calculate_distances(query_feature, features)
    top_k_indices = np.argsort(distances)[:top_k]
    top_k_images = [image_paths[i] for i in top_k_indices]

    print(f"Top {top_k} closest images:")
    for idx, img_path in enumerate(top_k_images):
        print(f"{idx + 1}: {img_path}")
    
    # Display top-k closest images in a new window
    display_top_k_images(top_k_images)
    
# Function to display top-k closest images in a new window
def display_top_k_images(image_paths):
    top_window = Toplevel()  # Create a new top-level window
    top_window.title("Top-K Closest Images")

    def on_close():
        root.destroy()

    top_window.protocol('WM_DELETE_WINDOW', on_close)
    
    for idx, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path)
            img.thumbnail((200, 200))  # Resize for display
            
            # Convert to PhotoImage to display in Tkinter
            img_tk = ImageTk.PhotoImage(img)
            
            label = Label(top_window, image=img_tk)
            label.image = img_tk  # Keep reference to prevent garbage collection
            label.grid(row=idx // 3, column=idx % 3, padx=10, pady=10)
        except Exception as e:
            print(f"Error displaying image {img_path}: {e}")

    # Button to close the window
    # close_button = Button(top_window, text="Close", command=root.destroy)

# Calculate distances from query image to dataset images
def calculate_distances(query_feature, features):
    distances = euclidean_distances(query_feature.reshape(1, -1), features).flatten()
    return distances

# Set up image preprocessing (resize, center crop, convert to tensor, normalize)
pp = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the images from a directory
image_dir = 'images/animals/animals/'
image_paths = glob.glob(os.path.join(image_dir, '**', '*.*'), recursive=True)
# image_paths = [img_path for img_path in image_paths]
# for img_path in image_paths:
#     print(img_path)
print(f'Loaded {len(image_paths)} images from {image_dir}')

def load_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = pp(img).unsqueeze(0)
    return img

# Load pre-trained ResNet model and remove the final classification layer
model = models.resnet50(weights='IMAGENET1K_V1')
model = torch.nn.Sequential(*(list(model.children())[:-1])) # Remove the final layer
model.eval()

# Extract features from the images
def extract_features(image_paths):
    features = []
    start_time = time.time()
    with torch.no_grad():
        for img_path in image_paths:
            img = load_image(img_path)
            feature = model(img).squeeze().numpy() # Extract features and flatten
            features.append(feature)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Feature extraction took {duration:.2f} seconds.")
    return np.array(features)

features = extract_features(image_paths)

# Use t-SNE for dimensionality reduction
def reduce_dimensions(features, n_components=2):
    tsne = TSNE(n_components=n_components, random_state=42)
    reduced_features = tsne.fit_transform(features)
    return reduced_features

# Generate 2D features for scatter plot
reduced_features_2d = reduce_dimensions(features, n_components=2)

# Generate 3D features for scatter plot
# reduced_features_3d = reduce_dimensions(features, n_components=3)

# 2D scatter plot of the reduced features
def plot_2d_features(features, image_paths):
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_facecolor('lightgrey')  # Set the axes background color
    fig.patch.set_facecolor('lightgrey')  # Set the figure background color

    scatter = ax.scatter(features[:, 0], features[:, 1], color='green', alpha=0.6)
    # plt.figure(figsize=(10,10))
    # plt.scatter(features[:, 0], features[:, 1], color='green', alpha=0.6)

    zoom_threshold = 200.0
    images_displayed = []# Text for displaying the zoom level
    zoom_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))

    # Load images and resize
    def load_and_rezie_images(image_paths, size=(50, 50)):
        images = []
        for path in image_paths:
            image = Image.open(path).resize(size)
            images.append(image)
        return images
    
    loaded_images = load_and_rezie_images(image_paths)

    # Variables to handle panning
    global press, x0, y0
    press = None
    x0 = None
    y0 = None

    def update_images():
        for img_display in images_displayed:
            img_display.remove()
        images_displayed.clear()

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        zoom_level = (xlim[1] - xlim[0]) * (ylim[1] - ylim[0])

        # Update zoom level text
        zoom_text.set_text(f'Zoom Level: {zoom_level:.2f}')

        if zoom_level < zoom_threshold:
            for i in range(len(features)):
                if (xlim[0] <= features[i, 0] <= xlim[1]) and (ylim[0] <= features[i, 1] <= ylim[1]):
                    imagebox = offsetbox.AnnotationBbox(
                        offsetbox.OffsetImage(loaded_images[i], zoom=1.0),
                        (features[i, 0], features[i, 1]),
                        frameon=False
                    )
                    ax.add_artist(imagebox)
                    images_displayed.append(imagebox)
        else:
            # ax.cla() # Clear the axis if the zoom level is not within the threshold
            ax.scatter(features[:, 0], features[:, 1], color='green', alpha=0.6)
            plt.title('2D Feature Representation of Images')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')

        plt.draw()
    
    def on_scroll(event):
    # Check if the event is within the axes
        if event.inaxes == ax:
            scale_factor = 0.95 if event.button == 'up' else 1.05  # Zoom in or out
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # Calculate the current center of the axes
            x_center = (xlim[1] + xlim[0]) / 2
            y_center = (ylim[1] + ylim[0]) / 2

            # Calculate new limites, keep the zoom centered
            new_xrange = (xlim[1] - xlim[0]) * scale_factor
            new_yrange = (ylim[1] - ylim[0]) * scale_factor

            ax.set_xlim([x_center - new_xrange / 2, x_center + new_xrange / 2])
            ax.set_ylim([y_center - new_yrange / 2, y_center + new_yrange / 2])
            
            update_images()
    
    def on_press(event):
        global press, x0, y0
        if event.button == 1:  # Left mouse button
            press = (event.xdata, event.ydata)

    def on_release(event):
        global press
        press = None  # Reset press on release

    def on_motion(event):
        global press, x0, y0
        if press is not None and event.inaxes == ax:
            dx = event.xdata - press[0]
            dy = event.ydata - press[1]

            # Only update if movement is significant
            threshold = 1
            if abs(dx) > threshold or abs(dy) > threshold:
                # Update axis limits
                ax.set_xlim(ax.get_xlim()[0] - dx, ax.get_xlim()[1] - dx)
                ax.set_ylim(ax.get_ylim()[0] - dy, ax.get_ylim()[1] - dy)

                # Update current position
                x0, y0 = event.xdata, event.ydata

                # Use canvas.draw_idle() for smoother updates
                ax.figure.canvas.draw_idle()
                # plt.draw()
        
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    
    plt.title('2D Feature Representation of Images')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

# 3D scatter plot of reduced features
def plot_3d_features(features, image_paths):
    # Create a figure and 3D axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D points
    ax.scatter(features[:, 0], features[:, 1], features[:, 2], color='orange', alpha=0.6)

    ax.set_title('3D Feature Representation of Images')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plt.show()

    # Scroll event to control zoom
    def on_scroll(event):
        # Get current limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()

        # Calculate the center of the current view
        x_center = (xlim[1] + xlim[0]) / 2
        y_center = (ylim[1] + ylim[0]) / 2
        z_center = (zlim[1] + zlim[0]) / 2

        # Zoom factor
        zoom_factor = 0.1

        # Adjust limits based on scroll direction
        if event.button == 'up':  # Zoom in
            ax.set_xlim([x_center - (x_center - xlim[0]) * (1 - zoom_factor),
                          x_center + (xlim[1] - x_center) * (1 - zoom_factor)])
            ax.set_ylim([y_center - (y_center - ylim[0]) * (1 - zoom_factor),
                          y_center + (ylim[1] - y_center) * (1 - zoom_factor)])
            ax.set_zlim([z_center - (z_center - zlim[0]) * (1 - zoom_factor),
                          z_center + (zlim[1] - z_center) * (1 - zoom_factor)])
        elif event.button == 'down':  # Zoom out
            ax.set_xlim([x_center - (x_center - xlim[0]) * (1 + zoom_factor),
                          x_center + (xlim[1] - x_center) * (1 + zoom_factor)])
            ax.set_ylim([y_center - (y_center - ylim[0]) * (1 + zoom_factor),
                          y_center + (ylim[1] - y_center) * (1 + zoom_factor)])
            ax.set_zlim([z_center - (z_center - zlim[0]) * (1 + zoom_factor),
                          z_center + (zlim[1] - z_center) * (1 + zoom_factor)])
        
        plt.draw()
    
    # Connect the scroll event
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    plt.show()

plot_2d_features(reduced_features_2d, image_paths)
# plot_3d_features(reduced_features_3d, image_paths)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the root window if you don’t want it
    upload_and_query_image(top_k=5)
    root.mainloop()  # Start the Tkinter main loop