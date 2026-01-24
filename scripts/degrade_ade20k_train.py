import cv2
import numpy as np
import os
from glob import glob

# ======================
# PATHS
# ======================
SRC_DIR = "/root/.cache/kagglehub/datasets/awsaf49/ade20k-dataset/versions/2/ADEChallengeData2016/images/training"
OUT_INPUT = "Dataset/train/input"
OUT_TARGET = "Dataset/train/target"

os.makedirs(OUT_INPUT, exist_ok=True)
os.makedirs(OUT_TARGET, exist_ok=True)

# ======================
# DEGRADATION FUNCTIONS
# ======================


def low_light(img, factor=0.4):
    return np.clip(img * factor, 0, 255).astype(np.uint8)


def gaussian_noise(img, sigma=15):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def gaussian_blur(img, k=5):
    return cv2.GaussianBlur(img, (k, k), 0)


def jpeg_compress(img, quality=30):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(enc, 1)


# ======================
# MAIN LOOP
# ======================
images = glob(os.path.join(SRC_DIR, "*.jpg"))
images = images[:600]   # use only first 600 images


idx = 0
for img_path in images:
    img = cv2.imread(img_path)
    if img is None:
        continue

    base_name = f"{idx:04d}"

    # save clean GT
    cv2.imwrite(os.path.join(OUT_TARGET, base_name + ".png"), img)

    # generate degradations
    degraded = [
        low_light(img),
        gaussian_noise(img),
        gaussian_blur(img),
        jpeg_compress(img)
    ]

    for i, dimg in enumerate(degraded):
        out_name = f"{base_name}_{i}.png"
        cv2.imwrite(os.path.join(OUT_INPUT, out_name), dimg)

    idx += 1
