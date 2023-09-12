import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# matrix = np.random.randint(2, size=(20, 20)).astype(bool)
# image =  Image.fromarray(matrix)
# image.save("test3.png", bits=1,optimize=True)

def flood_fill(image, x, y, new_color,searchcolor):

    rows, cols = 10 ,10
    #print(image.shape[:2])
    

    queue = [(x, y)]
    print(queue)
    while queue:
        condition = (image[x][y] ==  searchcolor)[0] and  (image[x][y] ==  searchcolor)[1] and (image[x][y] ==  searchcolor)[2] 
        print(condition)
        if 0 <= x < rows and 0 <= y < cols and condition:
            image[x][y] = new_color
            print(image[x][y])
            queue.append((x - 1, y+1))
            queue.append((x - 1, y-1))
            queue.append((x - 1, y))
            queue.append((x, y - 1))
        x, y = queue.pop(0)
    return image


im= cv2.imread("test.png")
img2=flood_fill(im, 5, 5, (255,255,0),(255,255,255))

cv2.imshow("image", img2)
cv2.waitKey()
cv2.destroyAllWindows()