import cv2
import numpy as np
from MyHybridImages import myHybridImages
from MyConvolution import convolve

image1_path = "/Users/mac/PycharmProjects/COMP3204CW1/data/einstein.bmp"
image2_path = "/Users/mac/PycharmProjects/COMP3204CW1/data/marilyn.bmp"

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# Iterate through all combinations of lowSigma and highSigma
for lowSigma in range(5, 11):
    for highSigma in range(5, 11):
        hybrid_image_np = myHybridImages(image1, lowSigma, image2, highSigma)
        hybrid_image_bgr = cv2.cvtColor(hybrid_image_np.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Display the image with the corresponding label
        window_name = f"low {lowSigma} high {highSigma}"
        cv2.imshow(window_name, hybrid_image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# If you want to save the last hybrid image
# cv2.imwrite("hybrid_output.jpg", hybrid_image_b
