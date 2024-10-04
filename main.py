import os
import glob
from PIL import Image
import torch
from torchvision import models, transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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
    with torch.no_grad():
        for img_path in image_paths:
            img = load_image(img_path)
            feature = model(img).squeeze().numpy() # Extract features and flatten
            features.append(feature)
    return np.array(features)

features = extract_features(image_paths)

# Use t-SNE for dimensionality reduction
def reduce_dimensions(features):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)
    return reduced_features

reduced_features = reduce_dimensions(features)

# Scatter plot of the reduced features
def plot_features(features, image_paths):
    plt.figure(figsize=(10,10))
    plt.scatter(features[:, 0], features[:, 1], color='blue', alpha=0.6)

    # for i, img_path in enumerate(image_paths):
    #     plt.annotate(os.path.basename(img_path), (features[i, 0], features[i, 1]))
    
    plt.title('2D Feature Representation of Images')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

plot_features(reduced_features, image_paths)