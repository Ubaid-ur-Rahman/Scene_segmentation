import cv2
import numpy as np

# Load image
img = cv2.imread("download1.jpg", cv2.IMREAD_GRAYSCALE)

# Apply edge detection
edges = cv2.Canny(img, 100, 200)  # 100 is the low threshold, 200 is the high threshold

# Apply binary thresholding
_, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

# Apply morphological opening to remove small noise
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Find horizontal and vertical lines
horizontal = cv2.reduce(opening, 1, cv2.REDUCE_SUM)
vertical = cv2.reduce(opening, 0, cv2.REDUCE_MAX)

# Find horizontal segments
h_lines = []
for i in range(horizontal.shape[0]):
    if (horizontal[i]).any() == 255:
        if not h_lines or h_lines[-1][1] != i - 1:
            h_lines.append([i, i])
        else:
            h_lines[-1][1] = i

# Find vertical segments
v_lines = []
for i in range(vertical.shape[0]):
    if (vertical[i]).any() == 255:
        if not v_lines or v_lines[-1][1] != i - 1:
            v_lines.append([i, i])
        else:
            v_lines[-1][1] = i

print(v_lines)
print(h_lines)
# Extract discriminant information for each segment
segments = []
for line in h_lines:
    segments.append({"type": "horizontal", "start": line[0], "length": line[1] - line[0] + 1,
                     "intensity": np.mean(img[line[0]:line[1]+1, :])})
for line in v_lines:
    segments.append({"type": "vertical", "start": line[0], "length": line[1] - line[0] + 1,
                     "intensity": np.mean(img[:, line[0]:line[1]+1])})

# Print segment information
for seg in segments:
    print(f"{seg['type']} segment starting at {seg['start']}, length {seg['length']}, intensity {seg['intensity']}")


# Assemble closest segments representing approximately a window
windows = []
for h_line in h_lines:
    for v_line in v_lines:
        if abs(h_line[0]-v_line[0]) <= 5 and abs(h_line[1]-v_line[1]) <= 5 and abs(h_line[1]-h_line[0]-v_line[1]+v_line[0]) <= 5:
            windows.append({"h_line": h_line, "v_line": v_line, "score": abs(h_line[0]-v_line[0])+abs(h_line[1]-v_line[1])+abs(h_line[1]-h_line[0]-v_line[1]+v_line[0])})

# Construct a new image containing only the estimated windows
result = np.zeros_like(img)
for window in windows:
    h1, h2 = window["h_line"]
    v1, v2 = window["v_line"]
    result[h1:h2+1, v1:v2+1] = img[h1:h2+1, v1:v2+1]

# Display the result
cv2.imshow("Estimated windows", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
