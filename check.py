import numpy as np
from MyConvolution import convolve
from MyHybridImages import myHybridImages
from MyHybridImages import makeGaussianKernel


data = np.load('kernels.npz')

kernel1 = data["kernel1"]
kernel2 = data["kernel2"]
kernel3 = data["kernel3"]



testKernel1 = makeGaussianKernel(1.0)
testKernel2 = makeGaussianKernel(2.0)
testKernel3 = makeGaussianKernel(5.5)


image = data["image"]
kernel = data["kernel"]
ans = data["ans"]
testans = convolve(image, kernel)

imageA = data["imageA"]
kernelA = data["kernelA"]
ansA = data["ansA"]
testansA = convolve(imageA, kernelA)


assert np.allclose(testKernel1, kernel1, atol=1e-6), f"kernel1 error"
assert np.allclose(testKernel2, kernel2, atol=1e-6), f"kernel2 error"
assert np.allclose(testKernel3, kernel3, atol=1e-6), f"kernel3 error"

assert np.allclose(ans, testans, atol=1e-6), f"convolve wrong"
assert np.allclose(ansA, testansA, atol=1e-6), f"convolveA wrong"
