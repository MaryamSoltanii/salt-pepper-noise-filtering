import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Read image (grayscale)
# --------------------------------------------------
img = cv2.imread("Lena.png", cv2.IMREAD_GRAYSCALE)

# --------------------------------------------------
# 2. Add Salt & Pepper noise (exactly 5% salt + 5% pepper)
# --------------------------------------------------
def add_salt_pepper_noise(image, amount=0.05):
    noisy = image.copy()
    num_pixels = image.size
    num_noisy = int(amount * num_pixels)

    # Half of the noisy pixels as Salt (white)
    num_salt = num_noisy // 2
    coords_salt = (
        np.random.randint(0, image.shape[0], num_salt),
        np.random.randint(0, image.shape[1], num_salt)
    )
    noisy[coords_salt] = 255

    # Half of the noisy pixels as Pepper (black)
    num_pepper = num_noisy - num_salt
    coords_pepper = (
        np.random.randint(0, image.shape[0], num_pepper),
        np.random.randint(0, image.shape[1], num_pepper)
    )
    noisy[coords_pepper] = 0

    return noisy


noisy = add_salt_pepper_noise(img, 0.05)

# --------------------------------------------------
# 3. Replicate padding (manual implementation)
# --------------------------------------------------
def replicate_pad(image, pad):
    h, w = image.shape
    padded = np.zeros((h + 2 * pad, w + 2 * pad), dtype=image.dtype)

    # Place original image in the center
    padded[pad:pad + h, pad:pad + w] = image

    # Top and bottom padding
    padded[:pad, pad:pad + w] = image[0:1, :]
    padded[pad + h:, pad:pad + w] = image[-1:, :]

    # Left and right padding
    padded[:, :pad] = padded[:, pad:pad + 1]
    padded[:, pad + w:] = padded[:, pad + w - 1:pad + w]

    return padded

# --------------------------------------------------
# 4. Manual nonlinear filters
# --------------------------------------------------
def min_filter(image, k=3):
    pad = k // 2
    padded = replicate_pad(image, pad)
    out = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i, j] = np.min(padded[i:i + k, j:j + k])

    return out


def max_filter(image, k=3):
    pad = k // 2
    padded = replicate_pad(image, pad)
    out = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i, j] = np.max(padded[i:i + k, j:j + k])

    return out


def median_filter(image, k=3):
    pad = k // 2
    padded = replicate_pad(image, pad)
    out = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i, j] = np.median(padded[i:i + k, j:j + k])

    return out

# --------------------------------------------------
# Apply filters
# --------------------------------------------------
min_img = min_filter(noisy)
max_img = max_filter(noisy)
med_img = median_filter(noisy)

# Combination of filters
min_then_max = max_filter(min_img)
max_then_min = min_filter(max_img)

# --------------------------------------------------
# Display results
# --------------------------------------------------
plt.figure(figsize=(15, 10))

plt.subplot(231), plt.imshow(noisy, cmap='gray'), plt.title('Noisy (Salt & Pepper 5% + 5%)')
plt.subplot(232), plt.imshow(min_img, cmap='gray'), plt.title('Min Filter')
plt.subplot(233), plt.imshow(max_img, cmap='gray'), plt.title('Max Filter')
plt.subplot(234), plt.imshow(med_img, cmap='gray'), plt.title('Median Filter')
plt.subplot(235), plt.imshow(min_then_max, cmap='gray'), plt.title('Min → Max')
plt.subplot(236), plt.imshow(max_then_min, cmap='gray'), plt.title('Max → Min')

plt.tight_layout()
plt.show()
