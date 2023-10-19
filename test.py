import cv2
import numpy as np
from MyHybridImages import myHybridImages
from MyConvolution import convolve

# Step 1: Load two images
image1_path = "/Users/mac/PycharmProjects/COMP3204CW1/data/cat.bmp"
image2_path = "/Users/mac/PycharmProjects/COMP3204CW1/data/dog.bmp"

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Convert images from BGR to RGB (OpenCV loads images in BGR format by default)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# Step 2: Decide sigma values
lowSigma = 5  # for low-pass filter
highSigma = 10 # for high-pass filter

# Step 3: Create hybrid image
hybrid_image_np = myHybridImages(image1, lowSigma, image2, highSigma)

# Convert the hybrid image from RGB to BGR for displaying using OpenCV
hybrid_image_bgr = cv2.cvtColor(hybrid_image_np.astype(np.uint8), cv2.COLOR_RGB2BGR)

# Display the hybrid image
cv2.imshow('Hybrid Image', hybrid_image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the hybrid image
# cv2.imwrite("hybrid_output.jpg", hybrid_image_bgr)
