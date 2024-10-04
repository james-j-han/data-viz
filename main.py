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
image_paths = [img_path for img_path in image_paths]
# for img_path in image_paths:
#     print(img_path)

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
# reduced_features_2d = reduce_dimensions(features, n_components=2)

# Generate 3D features for scatter plot
reduced_features_3d = reduce_dimensions(features, n_components=3)

# 2D scatter plot of the reduced features
def plot_2d_features(features, image_paths):
    plt.figure(figsize=(10,10))
    plt.scatter(features[:, 0], features[:, 1], color='blue', alpha=0.6)

    # for i, img_path in enumerate(image_paths):
    #     plt.annotate(os.path.basename(img_path), (features[i, 0], features[i, 1]))
    
    plt.title('2D Feature Representation of Images')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

# 3D scatter plot of reduced features
def plot_3d_features(features, image_paths):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(features[:, 0], features[:, 1], features[:, 2], color='blue', alpha=0.6)

    ax.set_title('3D Feature Representation of Images')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')

    # Enable interactive rotation and zoom
    # plt.ion() # Turn on interactive mode

    plt.show()

    # Keep the plot open and allow interaction
    # plt.pause(0.1) # Pause to allow for interaction

# plot_2d_features(reduced_features_2d, image_paths)
plot_3d_features(reduced_features_3d, image_paths)