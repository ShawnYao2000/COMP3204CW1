import numpy as np

def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve an image with a kernel assuming zero-padding of the image to handle the borders
    """
    # Get the image dimensions
    if len(image.shape) == 2:
        rows, cols = image.shape
        channels = 1
        image = image[:, :, np.newaxis]  # Convert grayscale to pseudo-color image for uniformity in processing
    else:
        rows, cols, channels = image.shape

    # Get kernel dimensions
    kheight, kwidth = kernel.shape
    pad_height = kheight // 2
    pad_width = kwidth // 2

    # Flip the kernel both horizontally and vertically
    flipped_kernel = np.zeros_like(kernel)
    for i in range(kheight):
        for j in range(kwidth):
            flipped_kernel[i, j] = kernel[kheight - 1 - i, kwidth - 1 - j]

    # Initialize the output image
    output = np.zeros((rows, cols, channels))

    # Pad the image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), 'constant')

    # Convolve for each channel
    for channel in range(channels):
        for i in range(rows):
            for j in range(cols):
                # Extract the region of interest
                region = padded_image[i:i + kheight, j:j + kwidth, channel]
                # Apply convolution (element-wise multiplication and sum)
                output[i, j, channel] = np.sum(region * flipped_kernel)

    # If grayscale, convert back from pseudo-color to grayscale
    if channels == 1:
        output = output[:, :, 0]

    #output = clip(output)

    return output


def clip(array: np.ndarray) -> np.ndarray:
    array[array < 0] = 0
    array[array > 255] = 255
    return array