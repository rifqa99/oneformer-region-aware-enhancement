import cv2
import numpy as np
import os
from glob import glob
import argparse


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


DEGRADATIONS = [
    low_light,
    gaussian_noise,
    gaussian_blur,
    jpeg_compress
]


def main(args):
    src_dir = os.path.join(
        args.ade_root,
        "images",
        args.split
    )

    out_input = os.path.join(args.out_root, args.split, "input")
    out_target = os.path.join(args.out_root, args.split, "target")

    os.makedirs(out_input, exist_ok=True)
    os.makedirs(out_target, exist_ok=True)

    images = sorted(glob(os.path.join(src_dir, "*.jpg")))
    images = images[:args.max_images]

    idx = 0
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue

        base_name = f"{idx:04d}"

        # Save clean GT
        cv2.imwrite(os.path.join(out_target, base_name + ".png"), img)

        # Generate degradations
        for i, degrade_fn in enumerate(DEGRADATIONS):
            dimg = degrade_fn(img)
            out_name = f"{base_name}_{i}.png"
            cv2.imwrite(os.path.join(out_input, out_name), dimg)

        idx += 1

    print(f"[✓] Generated {idx} samples for split '{args.split}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ade_root", type=str, required=True,
                        help="Path to ADE20K root directory")
    parser.add_argument("--out_root", type=str, default="Dataset",
                        help="Output dataset root")
    parser.add_argument("--split", type=str, choices=["training", "validation", "test"],
                        required=True)
    parser.add_argument("--max_images", type=int, default=100)

    args = parser.parse_args()
    main(args)
