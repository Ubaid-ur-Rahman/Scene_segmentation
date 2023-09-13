import numpy as np
import cv2

def convolution_smoothing(img, kernel_size=3, sigma=1):
    rows, cols = img.shape[:2]
    img = img.astype(float)
    smoothed_img = np.zeros((rows, cols), dtype=float)

    # Generate a Gaussian kernel with the given kernel size & sigma
    kernel = np.zeros((kernel_size, kernel_size), dtype=float)
    for i in range(-kernel_size//2, kernel_size//2+1):
        for j in range(-kernel_size//2, kernel_size//2+1):
            kernel[i+kernel_size//2, j+kernel_size//2] = (1/(2*np.pi*sigma**2)) * np.exp(-(i**2 + j**2)/(2*sigma**2))
    
    # Normalize the kernel so that its values sum to 1
    kernel /= np.sum(kernel)

    # Perform convolution
    for i in range(rows):
        for j in range(cols):
            window = img[max(0, i-kernel_size//2):min(rows, i+kernel_size//2+1),
                         max(0, j-kernel_size//2):min(cols, j+kernel_size//2+1)]
            window = np.expand_dims(window, axis=-1)
            smoothed_img[i, j] = np.sum(np.multiply(window, kernel))

    return smoothed_img

def canny_deriche_edge_detection(smoothed_img):
    rows, cols = smoothed_img.shape[:2]
    gradient_x = np.zeros((rows, cols), dtype=float)
    gradient_y = np.zeros((rows, cols), dtype=float)
    gradient_magnitude = np.zeros((rows, cols), dtype=float)
    gradient_direction = np.zeros((rows, cols), dtype=float)

    # calculate gradient in x and y direction
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            gradient_x[i, j] = smoothed_img[i+1, j-1] + 2 * smoothed_img[i+1, j] + smoothed_img[i+1, j+1] - smoothed_img[i-1, j-1] - 2 * smoothed_img[i-1, j] - smoothed_img[i-1, j+1]
            gradient_y[i, j] = smoothed_img[i-1, j+1] + 2 * smoothed_img[i, j+1] + smoothed_img[i+1, j+1] - smoothed_img[i-1, j-1] - 2 * smoothed_img[i, j-1] - smoothed_img[i+1, j-1]
    
    # calculate gradient magnitude and direction
    for i in range(rows):
        for j in range(cols):
            gradient_magnitude[i, j] = np.sqrt(gradient_x[i, j]**2 + gradient_y[i, j]**2)
            gradient_direction[i, j] = np.arctan2(gradient_y[i, j], gradient_x[i, j])
            
    return gradient_magnitude, gradient_direction

def non_maximum_suppression(magnitude, orientation):
    rows, cols = magnitude.shape[:2]
    NMS = np.zeros((rows, cols), dtype=float)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            theta = orientation[i, j]
            if (theta >= 0 and theta < 22.5) or (theta >= 157.5 and theta < 202.5):
                if (magnitude[i, j] > magnitude[i, j + 1]) and (magnitude[i, j] > magnitude[i, j - 1]):
                    NMS[i, j] = magnitude[i, j]
            elif (theta >= 22.5 and theta < 67.5) or (theta >= 202.5 and theta < 247.5):
                if (magnitude[i, j] > magnitude[i - 1, j + 1]) and (magnitude[i, j] > magnitude[i + 1, j - 1]):
                    NMS[i, j] = magnitude[i, j]
            elif (theta >= 67.5 and theta < 112.5) or (theta >= 247.5 and theta < 292.5):
                if (magnitude[i, j] > magnitude[i - 1, j]) and (magnitude[i, j] > magnitude[i + 1, j]):
                    NMS[i, j] = magnitude[i, j]
            elif (theta >= 112.5 and theta < 157.5) or (theta >= 292.5 and theta < 337.5):
                if (magnitude[i, j] > magnitude[i - 1, j - 1]) and (magnitude[i, j] > magnitude[i + 1, j + 1]):
                    NMS[i, j] = magnitude[i, j]

    return NMS


# Load the input image
img = cv2.imread('2.jpg')
# Apply the Deriche edge detector on the image
smoothed_image = convolution_smoothing(img)
magnitudes, direction = canny_deriche_edge_detection(smoothed_image)
edges = non_maximum_suppression(magnitudes, direction)
cv2.imshow("edges", edges)
cv2.waitKey()
cv2.destroyAllWindows()
