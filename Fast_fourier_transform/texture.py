import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from sklearn.cluster import KMeans

def extract_texture_features(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute the GLCM matrix
    glcm = greycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256,
                        symmetric=True, normed=True)
    
    # Compute the four properties of GLCM - contrast, correlation, energy, and homogeneity
    contrast = greycoprops(glcm, 'contrast').ravel()
    correlation = greycoprops(glcm, 'correlation').ravel()
    energy = greycoprops(glcm, 'energy').ravel()
    homogeneity = greycoprops(glcm, 'homogeneity').ravel()
    
    # Concatenate the four feature vectors
    feature_vector = np.hstack([contrast, correlation, energy, homogeneity])
    
    return feature_vector


# Define the number of clusters (groups)
num_clusters = 2

# Load the wall and leaf images and extract texture features from them
wall_images = ['2.jpg', '3.jpg', '5.jpg', '6.jpg', '9.jpg', '11.jpg', '14.jpg']
leaf_images = ['1.jpg', '4.jpg', '7.jpg', '8.jpg', '10.jpg']

wall_features = []
for image_path in wall_images:
    image = cv2.imread(image_path)
    features = extract_texture_features(image)
    wall_features.append(features)
    
leaf_features = []
for image_path in leaf_images:
    image = cv2.imread(image_path)
    features = extract_texture_features(image)
    leaf_features.append(features)

# Perform K-means clustering on the texture feature vectors for each group
kmeans_wall = KMeans(n_clusters=num_clusters, random_state=0).fit(wall_features)
kmeans_leaf = KMeans(n_clusters=num_clusters, random_state=0).fit(leaf_features)

def identify_group(image_path):
    # Load the input image and extract texture features
    image = cv2.imread(image_path)
    features = extract_texture_features(image)
    
    # Predict the cluster for the input image using the K-means models
    wall_cluster = kmeans_wall.predict([features])[0]
    leaf_cluster = kmeans_leaf.predict([features])[0]
    
    # Determine which group the input image belongs to based on the cluster predictions
    if wall_cluster == leaf_cluster:
        return "Unknown"
    elif wall_cluster < leaf_cluster:
        return "Wall"
    else:
        return "Leaf"
    

group = identify_group("15.jpg")
print(group)