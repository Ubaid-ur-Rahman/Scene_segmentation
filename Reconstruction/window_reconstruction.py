import cv2
import numpy as np

# Load the image
img = cv2.imread('download1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply edge detection using the Canny algorithm
edges = cv2.Canny(gray, 100, 200)

# Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
gray = cv2.bitwise_not(gray)
bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

cv2.imshow("horizontal", bw)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create the images that will use to extract the horizontal and vertical lines
horizontal = np.copy(bw)
vertical = np.copy(bw)

# Specify size on horizontal axis
cols = horizontal.shape[1]
horizontal_size = cols // 30
# Create structure element for extracting horizontal lines through morphology operations
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
# Apply morphology operations
horizontal = cv2.erode(horizontal, horizontalStructure)
horizontal = cv2.dilate(horizontal, horizontalStructure)
# Show extracted horizontal lines
cv2.imshow("horizontal", horizontal)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Specify size on vertical axis
rows = vertical.shape[0]
verticalsize = rows // 30
# Create structure element for extracting vertical lines through morphology operations
verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
# Apply morphology operations
vertical = cv2.erode(vertical, verticalStructure)
vertical = cv2.dilate(vertical, verticalStructure)
# Show extracted vertical lines
cv2.imshow("vertical", vertical)
cv2.waitKey(0)
cv2.destroyAllWindows()


def concatenate_images(vertical_lines, horizontal_lines):
    # Ensure both images have the same size
    if vertical_lines.shape != horizontal_lines.shape:
        raise ValueError("Images must have the same size")
    # Create an empty image for the concatenated result
    result = np.zeros_like(vertical_lines)
    # Combine the vertical and horizontal lines
    result[vertical_lines != 255] = 255
    result[horizontal_lines != 255] = 255
    return result


def get_vertical_lines(image):
    # Create an empty list for the line coordinates
    lines = []
    # Find the vertical edges using the Sobel operator
    edges = cv2.Sobel(image, cv2.CV_8U, 1, 0)
    # Find the contours of the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Loop through each contour
    for contour in contours:
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)
        # Only keep contours that are tall and thin enough to be lines
        if h > 5 * w:
            # Add the line coordinates to the list
            lines.append((x, y, x + w, y + h))
    return lines


def get_horizontal_lines(image):
    # Create an empty list for the line coordinates
    lines = []
    # Find the horizontal edges using the Sobel operator
    edges = cv2.Sobel(image, cv2.CV_8U, 0, 1)
    # Find the contours of the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Loop through each contour
    for contour in contours:
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)
        # Only keep contours that are wide and short enough to be lines
        if w > 5 * h:
            # Add the line coordinates to the list
            lines.append((x, y, x + w, y + h))
    return lines


horizontal_points=get_horizontal_lines(horizontal)
vertical_points=get_vertical_lines(vertical)
print(horizontal)
print("now we will generate vertical")
print(vertical)


# Inverse vertical image
vertical = cv2.bitwise_not(vertical)
edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
# Step 2
kernel = np.ones((2, 2), np.uint8)
edges = cv2.dilate(edges, kernel)
# Step 3
smooth = np.copy(vertical)
# Step 4
smooth = cv2.blur(smooth, (2, 2))
# Step 5
(rows, cols) = np.where(edges != 0)
vertical[rows, cols] = smooth[rows, cols]
# Show final result
cv2.imshow("Recreated vertical lines", vertical)
cv2.waitKey(0)
cv2.destroyAllWindows()


horizontal = cv2.bitwise_not(horizontal)
edges = cv2.adaptiveThreshold(horizontal, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
# Step 2
kernel = np.ones((2, 2), np.uint8)
edges = cv2.dilate(edges, kernel)
# Step 3
smooth = np.copy(horizontal)
# Step 4
smooth = cv2.blur(smooth, (2, 2))
# Step 5
(rows, cols) = np.where(edges != 0)
horizontal[rows, cols] = smooth[rows, cols] 


result= concatenate_images(vertical,horizontal)
# Show final result
cv2.imshow("final", result)
cv2.waitKey(0)
cv2.destroyAllWindows()


