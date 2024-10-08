from io import BytesIO
import os
import glob
from PIL import Image
from PIL import ImageTk
import time
import torch
from torchvision import models, transforms
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("TkAgg") # IMPORTANT in this order for tkinter to run properly
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.offsetbox as offsetbox
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, Toplevel, Label, Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics.pairwise import euclidean_distances
import requests
import openai

def get_api_key(file_path='api_key.txt'):
    with open(file_path, 'r') as file:
        return file.read().strip()

client = openai.OpenAI(api_key=get_api_key())

# Function to generate image from text input
def generate_image_from_text(text):
    try:
        response = client.images.generate(
            model='dall-e-3',
            prompt=text,
            size='1024x1024',
            n=1,
            quality='standard'
        )
        image_url = response.data[0].url
        print('Generated Image URL:', image_url)
        return image_url
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

# Global variables
features = []
existing_features_2d = None
existing_features_3d = None
fig2d, ax2 = None, None
fig3d, ax3 = None, None

# Update the command in your button to include text input
def on_query_button_click(prompt):
    image_url = generate_image_from_text(prompt)
    # print(image_url)

    if image_url:
        # Load the generated image
        # new_image = Image.open(requests.get(image_url, stream=True).raw)
        # new_image = new_image.convert('RGB')  # Ensure it's RGB format
        # new_image = pp(new_image).unsqueeze(0)  # Preprocess for the model
        
        # Extract features
        print('Image URL from generate image: ', image_url)
        updated_features = extract_features([image_url]) # Need to pass as array

        # Append new features to existing features
        # features.append(new_feature)

        # Update the plots with the new point
        update_plots(updated_features)

        # Display the generated image in a new window
        display_image_window(image_url)

def update_plots(new_feature):
    # global existing_features_2d, existing_features_3d

    # Append new feature for 2D and 3D plots
    existing_features_2d = reduce_dimensions(new_feature, n_components=2)
    existing_features_3d = reduce_dimensions(new_feature, n_components=3)

    # Update 2D plot
    ax2.clear()  # Clear previous 2D scatter plot
    ax2.scatter(existing_features_2d[:, 0], existing_features_2d[:, 1], color='green', alpha=0.6)
    ax2.scatter(existing_features_2d[-1, 0], existing_features_2d[-1, 1], color='red', s=100, label='New Image')  # New point in red
    ax2.legend()
    fig2d.canvas.draw_idle()  # Redraw 2D plot

    # Update 3D plot
    ax3.clear()  # Clear previous 3D scatter plot
    ax3.scatter(existing_features_3d[:, 0], existing_features_3d[:, 1], existing_features_3d[:, 2], color='orange', alpha=0.6)
    ax3.scatter(existing_features_3d[-1, 0], existing_features_3d[-1, 1], existing_features_3d[-1, 2], color='red', s=100, label='New Image')  # New point in red
    ax3.legend()
    fig3d.canvas.draw_idle()  # Redraw 3D plot

def display_image_window(image_url):
    new_window = Toplevel()  # Create a new top-level window
    new_window.title("Generated Image")

    def on_close():
        new_window.destroy()

    new_window.protocol('WM_DELETE_WINDOW', on_close)

    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        img_data = Image.open(BytesIO(response.content))
        img_data.thumbnail((800, 800))  # Resize for display, if needed

        # Convert to PhotoImage to display in Tkinter
        img_tk = ImageTk.PhotoImage(img_data)

        label = Label(new_window, image=img_tk)
        label.image = img_tk  # Keep reference to avoid garbage collection
        label.pack()

    except Exception as e:
        print(f"Error loading image from URL: {e}")

    # new_window.mainloop()

# Function to upload a single image
def upload_and_query_image(top_k=5):
    print(f"Selected integer: {top_k}")
    k = int(top_k)
    img_path = filedialog.askopenfilename(title="Select an image to query")
    if not img_path:
        print("No image selected")
        # root.destroy()
        return None

    print(f'Uploaded image: {img_path}')
    
    # Extract features from the uploaded image
    query_img = load_image(img_path)
    with torch.no_grad():
        query_feature = model(query_img).squeeze().numpy()

    # Calculate distances and find top-k closest images
    distances = calculate_distances(query_feature, features)
    top_k_indices = np.argsort(distances)[:k]
    top_k_images = [image_paths[i] for i in top_k_indices]

    print(f"Top {k} closest images:")
    for idx, img_path in enumerate(top_k_images):
        print(f"{idx + 1}: {img_path}")
    
    # Display top-k closest images in a new window
    display_top_k_images(top_k_images, k)
    
# Function to display top-k closest images in a new window
def display_top_k_images(image_paths, k):
    top_window = Toplevel()  # Create a new top-level window
    top_window.title(f"Top-{k} Closest Images")

    def on_close():
        top_window.destroy()

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
print(f'Loaded {len(image_paths)} images from {image_dir}')

def load_image(img_path_or_url):
    if not img_path_or_url:  # Check if the path or URL is empty
        raise ValueError("No valid image path or URL provided")
    # Handle image path or image url
    url = img_path_or_url.lower().strip()
    # print(url)
    if url.startswith('http://') or url.startswith('https://'):
        print(img_path_or_url)
        response = requests.get(img_path_or_url, stream=True)
        response.raise_for_status()  # Check if the request was successful
        # img = Image.open(BytesIO(response.content)).convert('RGB')
        img = Image.open(response.raw)
        print(img)
    else:
        img = Image.open(img_path_or_url)

    img.convert('RGB')
    img_tensor = pp(img).unsqueeze(0)
    return img_tensor

# Load pre-trained ResNet model and remove the final classification layer
model = models.resnet50(weights='IMAGENET1K_V1')
model = torch.nn.Sequential(*(list(model.children())[:-1])) # Remove the final layer
model.eval()

# Extract features from the images
def extract_features(image_paths):
    # print('Extract Features: ', image_paths)
    # features = []
    start_time = time.time()
    with torch.no_grad():
        for img_path in image_paths:
            img_tensor = load_image(img_path)
            feature = model(img_tensor).squeeze().numpy() # Extract features and flatten
            features.append(feature)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Feature extraction took {duration:.2f} seconds.")
    # return np.array(features)
    return features

features = extract_features(image_paths)

# Use t-SNE for dimensionality reduction
def reduce_dimensions(features, n_components=2):
    features_to_reduce = convert_to_nparray(features)
    tsne = TSNE(n_components=n_components, random_state=42)
    reduced_features = tsne.fit_transform(features_to_reduce)
    return reduced_features

def convert_to_nparray(array_to_convert):
    return np.array(array_to_convert)

# Generate 2D features for scatter plot
reduced_features_2d = reduce_dimensions(features, n_components=2)

# Generate 3D features for scatter plot
reduced_features_3d = reduce_dimensions(features, n_components=3)

# 2D scatter plot of the reduced features
def plot_2d_features(features, image_paths, root):
    # Create a frame in Tkinter window to embed the plot
    # plot_frame = tk.Frame(root)
    # plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    global fig2d, ax2
    fig2d, ax2 = plt.subplots(figsize=(10, 10))

    # ax.set_facecolor('lightgrey')  # Set the axes background color
    # fig.patch.set_facecolor('lightgrey')  # Set the figure background color

    scatter = ax2.scatter(features[:, 0], features[:, 1], color='green', alpha=0.6)
    
    # Embed the figure in Tkinter window
    # canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    # canvas.draw()
    # canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    zoom_threshold = 200.0
    images_displayed = []# Text for displaying the zoom level
    zoom_text = ax2.text(0.05, 0.95, '', transform=ax2.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))

    # Load images and resize
    def load_and_resize_images(image_paths, size=(50, 50)):
        images = []
        for path in image_paths:
            image = Image.open(path).resize(size)
            images.append(image)
        return images
    
    loaded_images = load_and_resize_images(image_paths)

    # Variables to handle panning
    global press
    press = None
    # global press, x0, y0
    # press = None
    # x0 = None
    # y0 = None

    def update_images():
        for img_display in images_displayed:
            img_display.remove()
        images_displayed.clear()

        xlim = ax2.get_xlim()
        ylim = ax2.get_ylim()

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
                    ax2.add_artist(imagebox)
                    images_displayed.append(imagebox)
        else:
            # ax.cla() # Clear the axis if the zoom level is not within the threshold
            ax2.scatter(features[:, 0], features[:, 1], color='green', alpha=0.6)
            plt.title('2D Feature Representation of Images')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')

        fig2d.canvas.draw_idle()
        # plt.draw()
        # canvas.draw()
    
    def on_scroll(event):
    # Check if the event is within the axes
        if event.inaxes == ax2:
            scale_factor = 0.95 if event.button == 'up' else 1.05  # Zoom in or out
            xlim = ax2.get_xlim()
            ylim = ax2.get_ylim()

            # Calculate the current center of the axes
            x_center = (xlim[1] + xlim[0]) / 2
            y_center = (ylim[1] + ylim[0]) / 2

            # Calculate new limites, keep the zoom centered
            new_xrange = (xlim[1] - xlim[0]) * scale_factor
            new_yrange = (ylim[1] - ylim[0]) * scale_factor

            ax2.set_xlim([x_center - new_xrange / 2, x_center + new_xrange / 2])
            ax2.set_ylim([y_center - new_yrange / 2, y_center + new_yrange / 2])
            
            update_images()
    
    def on_press(event):
        global press
        if event.button == 1:  # Left mouse button
            press = (event.xdata, event.ydata)

    def on_release(event):
        global press
        press = None  # Reset press on release

    def on_motion(event):
        global press
        if press is not None and event.inaxes == ax2:
            dx = event.xdata - press[0]
            dy = event.ydata - press[1]

            xlim = ax2.get_xlim()
            ylim = ax2.get_ylim()

            # Only update if movement is significant
            # threshold = 1
            # if abs(dx) > threshold or abs(dy) > threshold:
            # Update axis limits
            ax2.set_xlim(xlim[0] - dx, xlim[1] - dx)
            ax2.set_ylim(ylim[0] - dy, ylim[1] - dy)

            # Update current position
            # x0, y0 = event.xdata, event.ydata

            # Use canvas.draw_idle() for smoother updates
            # ax.figure.canvas.draw_idle()
            # canvas.draw_idle()
            fig2d.canvas.draw_idle()
        
    fig2d.canvas.mpl_connect('scroll_event', on_scroll)
    fig2d.canvas.mpl_connect('button_press_event', on_press)
    fig2d.canvas.mpl_connect('button_release_event', on_release)
    fig2d.canvas.mpl_connect('motion_notify_event', on_motion)
    
    plt.title('2D Feature Representation of Images')
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.show()

    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig2d, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# 3D scatter plot of reduced features
def plot_3d_features(features, image_paths, root):
    # Create a figure and 3D axis
    global fig3d, ax3
    fig3d = plt.figure(figsize=(10, 10))
    ax3 = fig3d.add_subplot(111, projection='3d')

    # Plot 3D points
    ax3.scatter(features[:, 0], features[:, 1], features[:, 2], color='orange', alpha=0.6)

    ax3.set_title('3D Feature Representation of Images')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    # plt.show()

    # Scroll event to control zoom
    def on_scroll(event):
        # Get current limits
        xlim = ax3.get_xlim()
        ylim = ax3.get_ylim()
        zlim = ax3.get_zlim()

        # Calculate the center of the current view
        x_center = (xlim[1] + xlim[0]) / 2
        y_center = (ylim[1] + ylim[0]) / 2
        z_center = (zlim[1] + zlim[0]) / 2

        # Zoom factor
        zoom_factor = 0.1

        # Adjust limits based on scroll direction
        if event.button == 'up':  # Zoom in
            ax3.set_xlim([x_center - (x_center - xlim[0]) * (1 - zoom_factor),
                          x_center + (xlim[1] - x_center) * (1 - zoom_factor)])
            ax3.set_ylim([y_center - (y_center - ylim[0]) * (1 - zoom_factor),
                          y_center + (ylim[1] - y_center) * (1 - zoom_factor)])
            ax3.set_zlim([z_center - (z_center - zlim[0]) * (1 - zoom_factor),
                          z_center + (zlim[1] - z_center) * (1 - zoom_factor)])
        elif event.button == 'down':  # Zoom out
            ax3.set_xlim([x_center - (x_center - xlim[0]) * (1 + zoom_factor),
                          x_center + (xlim[1] - x_center) * (1 + zoom_factor)])
            ax3.set_ylim([y_center - (y_center - ylim[0]) * (1 + zoom_factor),
                          y_center + (ylim[1] - y_center) * (1 + zoom_factor)])
            ax3.set_zlim([z_center - (z_center - zlim[0]) * (1 + zoom_factor),
                          z_center + (zlim[1] - z_center) * (1 + zoom_factor)])
        
        # plt.draw()
        fig3d.canvas.draw_idle()
    
    # Connect the scroll event
    fig3d.canvas.mpl_connect('scroll_event', on_scroll)

    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig3d, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # plt.show()

# plot_2d_features(reduced_features_2d, image_paths)
# plot_3d_features(reduced_features_3d, image_paths)

if __name__ == "__main__":
    root = tk.Tk()
    root.title('Data Viz')

    # Define a function that will be called when the window is closed
    def on_close():
        print("Window closed")  # You can perform any cleanup here
        root.quit() # Stop the Tkinter main loop
        root.destroy()  # Close the Tkinter window

    # Bind the window close event (WM_DELETE_WINDOW is triggered when you click the "X")
    root.protocol("WM_DELETE_WINDOW", on_close)

    # Create a frame for the 2D and 3D plots
    plot_frame = tk.Frame(root)
    plot_frame.pack(fill=tk.BOTH, expand=True)

    # Create a dropdown (Combobox) with integer values
    int_values = list(range(1, 20))  # List of integers from 1 to 10
    dropdown = ttk.Combobox(root, values=int_values, state='readonly')  # Set state to readonly
    dropdown.pack(side=tk.LEFT, padx=(0, 2), pady=10)  # Pack the dropdown on the left side

    # root.withdraw()  # Hide the root window if you don’t want it
    # Call the plot function to display the scatter plot in tkinter window
    plot_2d_features(reduced_features_2d, image_paths, plot_frame)
    plot_3d_features(reduced_features_3d, image_paths, plot_frame)
    query_by_image_button = tk.Button(
        root,
        text="Upload and Query Image",
        command=lambda: upload_and_query_image(dropdown.get())) # Pass the selected value
    # query_by_image_button.pack(pady=10)
    query_by_image_button.pack(side=tk.LEFT)

    # Create a text entry field for text queries
    query_by_text_field = tk.Entry(root, width=30)
    query_by_text_field.pack(side=tk.LEFT, padx=(5, 0))  # Add some padding to the left

    query_by_text_button = tk.Button(
        root,
        text='Generate',
        command=lambda: on_query_button_click(query_by_text_field.get())
    )
    query_by_text_button.pack(side=tk.LEFT, padx=(5, 0))

    # upload_and_query_image(top_k=5)
    root.mainloop()  # Start the Tkinter main loop