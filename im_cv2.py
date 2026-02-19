import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ---------------------------------------------------
# 1. Load Image
# ---------------------------------------------------
img = cv2.imread("test.jpg")   # change path
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to float for metrics
img_float = img.astype(np.float32)

# ---------------------------------------------------
# 2. Add Synthetic Noise (for testing filters)
# ---------------------------------------------------
noise = np.random.normal(0, 25, img.shape).astype(np.float32)
noisy_img = img_float + noise
noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

# ---------------------------------------------------
# 3. Apply Filters
# ---------------------------------------------------

# Gaussian Filter
gaussian = cv2.GaussianBlur(noisy_img, (5,5), 1.5)

# Median Filter
median = cv2.medianBlur(noisy_img, 5)

# Bilateral Filter
bilateral = cv2.bilateralFilter(noisy_img, 9, 75, 75)

# ---------------------------------------------------
# 4. Compute PSNR
# ---------------------------------------------------
psnr_gaussian = psnr(img, gaussian, data_range=255)
psnr_median   = psnr(img, median, data_range=255)
psnr_bilateral= psnr(img, bilateral, data_range=255)

# ---------------------------------------------------
# 5. Compute SSIM
# ---------------------------------------------------
ssim_gaussian = ssim(img, gaussian, channel_axis=2, data_range=255)
ssim_median   = ssim(img, median, channel_axis=2, data_range=255)
ssim_bilateral= ssim(img, bilateral, channel_axis=2, data_range=255)

# ---------------------------------------------------
# 6. Print Metrics
# ---------------------------------------------------
print("PSNR Values:")
print("Gaussian :", psnr_gaussian)
print("Median   :", psnr_median)
print("Bilateral:", psnr_bilateral)

print("\nSSIM Values:")
print("Gaussian :", ssim_gaussian)
print("Median   :", ssim_median)
print("Bilateral:", ssim_bilateral)

# ---------------------------------------------------
# 7. Display Images
# ---------------------------------------------------
titles = [
    "Original",
    "Noisy",
    "Gaussian",
    "Median",
    "Bilateral"
]

images = [
    img,
    noisy_img,
    gaussian,
    median,
    bilateral
]

plt.figure(figsize=(15,8))

for i in range(len(images)):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
