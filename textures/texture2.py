import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_texture_features(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute the 2D FFT of the image
    fft = np.fft.fft2(gray)
    
    # Shift the zero frequency component to the center of the spectrum
    fft_shifted = np.fft.fftshift(fft)
    
    # Compute the power spectrum
    magnitude_spectrum = 20 * np.log(np.abs(fft_shifted))
    
    # Crop the power spectrum to eliminate the DC component and high frequencies
    crop_size = 128
    cy, cx = fft_shifted.shape[:2]
    start_x, start_y = int(cx/2-crop_size/2), int(cy/2-crop_size/2)
    end_x, end_y = start_x + crop_size, start_y + crop_size
    cropped_spectrum = magnitude_spectrum[start_y:end_y, start_x:end_x]
    
    # Flatten the cropped spectrum to a 1D feature vector
    features = cropped_spectrum.flatten()
    
    return features


def train_texture_model(wall_images, wave_images):
    # Extract texture features from the wall images
    wall_features = []
    for image_path in wall_images:
        image = cv2.imread(image_path)
        features = extract_texture_features(image)
        wall_features.append(features)
    
    # Extract texture features from the wave images
    wave_features = []
    for image_path in wave_images:
        image = cv2.imread(image_path)
        features = extract_texture_features(image)
        wave_features.append(features)
    
    # Compute the mean feature vector for each group
    wall_mean = np.mean(wall_features, axis=0)
    wave_mean = np.mean(wave_features, axis=0)
    
    # Return the mean feature vectors for each group
    return wall_mean, wave_mean

def identify_group(image_path, wall_mean, wave_mean):
    # Load the input image and extract its texture features
    image = cv2.imread(image_path)
    features = extract_texture_features(image)
    
    # Compute the Euclidean distance between the input image's feature vector and the mean feature vectors for each group
    wall_dist = np.linalg.norm(features - wall_mean)
    wave_dist = np.linalg.norm(features - wave_mean)
    
    # Determine which group the input image belongs to based on the distance to the mean feature vectors
    if wall_dist < wave_dist:
        return "Wall"
    else:
        return "waves"
    
wall_images = ['2.jpg', '3.jpg', '5.jpg', '6.jpg', '9.jpg', '11.jpg', '14.jpg']
wave_images = ['1.jpg', '4.jpg', '7.jpg', '8.jpg', '10.jpg', '12.jpg', '13.jpg', '17.jpg']

wall_mean, wave_mean=train_texture_model(wall_images, wave_images)
group = identify_group("15.jpg", wall_mean, wave_mean)
print(group)