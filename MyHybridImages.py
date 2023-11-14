import numpy as np
from MyConvolution import convolve, clip


def makeGaussianKernel(sigma: float) -> np.ndarray:
    """
    Create a 2D gaussian kernel with standard deviation sigma.
    """
    # Calculate the size of the kernel
    size = int(8.0 * sigma + 1.0)
    if size % 2 == 0:
        size += 1

    # Create the kernel
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) *
                     np.exp(-((x - (size // 2)) ** 2 + (y - (size // 2)) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )

    # Normalize the kernel
    kernel /= np.sum(kernel)

    return kernel


def myHybridImages(lowImage: np.ndarray, lowSigma: float,
                   highImage: np.ndarray, highSigma: float) -> np.ndarray:
    """
    Create hybrid images by combining a low-pass and high-pass filtered pair.
    """
    # Generate the Gaussian kernels
    low_kernel = makeGaussianKernel(lowSigma)
    high_kernel = makeGaussianKernel(highSigma)

    # Low-pass filter the lowImage
    low_pass_lowImage = convolve(lowImage, low_kernel)

    # Low-pass filter the highImage and then create the high-pass filtered image
    low_pass_highImage = convolve(highImage, high_kernel)
    high_pass_highImage = highImage - low_pass_highImage

    # Create the hybrid image
    hybrid_image = low_pass_lowImage + high_pass_highImage

    hybrid_image = clip(hybrid_image)

    return hybrid_image
