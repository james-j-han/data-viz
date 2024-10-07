import os
import glob
from PIL import Image
import time
import torch
from torchvision import models, transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.offsetbox as offsetbox

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

    zoom_threshold = 100.0
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
            scale_factor = 1.2 if event.button == 'up' else 0.8  # Zoom in or out
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
            x0, y0 = event.xdata, event.ydata  # Store initial position

    def on_release(event):
        global press
        press = None  # Reset press on release

    def on_motion(event):
        global press, x0, y0
        if press is not None and event.inaxes == ax:
            dx = event.xdata - x0
            dy = event.ydata - y0

            if abs(dx) > 1 or abs(dy) > 1: # Only update if movement is significant
            # Update axis limits
                ax.set_xlim(ax.get_xlim()[0] - dx, ax.get_xlim()[1] - dx)
                ax.set_ylim(ax.get_ylim()[0] - dy, ax.get_ylim()[1] - dy)

                x0, y0 = event.xdata, event.ydata  # Update current position
                plt.draw()
        
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
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(features[:, 0], features[:, 1], features[:, 2], color='blue', alpha=0.6)
    ax.autoscale(False)

    # Force the view to look like a 2D plot by setting elevation and azimuth
    # ax.view_init(elev=90, azim=-90)

    ax.set_title('3D Feature Representation of Images')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set zoom threshold
    zoom_threshold = 10000.0 # Adjust this value to set how close you need to zoom in to show images
    images_displayed = [] # Store the images displayed to prevent re-adding
    zoom_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes) # Text for zoom level

    # Preload and resize images
    def preprocess_image(img_path):
        img = Image.open(img_path).convert('RGB').resize((50, 50))
        return np.array(img) / 255.0
    
    images = [preprocess_image(img_path) for img_path in image_paths]
    
    # Callback to check the zoom level and display images
    def update_images():
        for img in images_displayed:
            if img in ax.collections:
                img.remove()
        images_displayed.clear()

        # Check the current limits to determine zoom level
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()

        # Set the initial view so the z-axis is towards the viewer
        # ax.view_init(elev=90, azim=0)  # Elevation of 90 degrees and azimuth of 0 degrees

        # Calculate the current zoom level
        zoom_level = (xlim[1] - xlim[0]) * (ylim[1] - ylim[0]) * (zlim[1] - zlim[0])

        # Update zoom level text
        zoom_text.set_text(f'Zoom Level: {zoom_level:.2f}')

        if zoom_level < zoom_threshold:
            # Filter features within the current limits
            for i, (x, y, z) in enumerate(features):
                # x, y, z = features[i]
                # img_path = image_paths[i]
                if (xlim[0] <= x <= xlim[1]) and (ylim[0] <= y <= ylim[1]) and (zlim[0] <= z <= zlim[1]):
                    img_array = images[i]
                
                    # Dynamically scale image size based on zoom level
                    img_size = max(50, 100 / (zoom_level ** 0.5))

                    x_mesh, y_mesh = np.meshgrid(
                        np.linspace(-img_size / 200, img_size / 200, 20),
                        np.linspace(-img_size / 200, img_size / 200, 20)
                    )
                    z_mesh = np.zeros_like(x_mesh) + z
                    
                    # Plot the image as a surface
                    surf = ax.plot_surface(x_mesh + x, y_mesh + y, z_mesh, rstride=1, cstride=1, facecolors=img_array, alpha=1, antialiased=False)
                    images_displayed.append(surf)
    
    def on_zoom(event):
        if event.inaxes == ax:
            update_images()
            plt.draw()

    fig.canvas.mpl_connect('button_release_event', on_zoom)
    # fig.canvas.mpl_connect('motion_notify_event', on_zoom)
    plt.show()

plot_2d_features(reduced_features_2d, image_paths)
# plot_3d_features(reduced_features_3d, image_paths)