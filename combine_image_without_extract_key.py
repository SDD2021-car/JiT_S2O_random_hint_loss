import os
import numpy as np
import cv2
import argparse
from multiprocessing import Pool


def is_image_file(fn: str) -> bool:
    ext = os.path.splitext(fn)[1].lower()
    return ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]


def resize_and_concatenate_three(path_A, path_B, path_C, out_path):
    im_A = cv2.imread(path_A, 1)
    im_B = cv2.imread(path_B, 1)
    im_C = cv2.imread(path_C, 1)

    if im_A is None or im_B is None or im_C is None:
        print(f"[WARN] Read failed:\n  A={path_A}\n  B={path_B}\n  C={path_C}")
        return

    hA, wA = im_A.shape[:2]
    hB, wB = im_B.shape[:2]
    hC, wC = im_C.shape[:2]

    target_h = min(hA, hB, hC)
    target_w = min(wA, wB, wC)

    im_A = cv2.resize(im_A, (target_w, target_h))
    im_B = cv2.resize(im_B, (target_w, target_h))
    im_C = cv2.resize(im_C, (target_w, target_h))

    im_ABC = np.concatenate([im_A, im_B, im_C], axis=1)
    cv2.imwrite(out_path, im_ABC)


parser = argparse.ArgumentParser("combine three images by same filename")
parser.add_argument('--fold_A', default="/NAS_data/yjy/ColorS2O_random_hint_concat/hint_output/dot/hints/overlay", type=str)
parser.add_argument('--fold_B', default="/NAS_data/yjy/ColorS2O_random_hint_concat/hint_output/dot/hints/color", type=str)
parser.add_argument('--fold_C', default="/NAS_data/hjf/JiTcolor/outputs/SAR2Opt/caJiT_CP/round4/noLoss_noHintsDropout_dot_concat/heun-steps50-cfg1.0-interval0.0-1.0-image50000-res512", type=str)
parser.add_argument('--fold_ABC', default="/NAS_data/yjy/ColorS2O_random_hint_concat/combine", type=str)
parser.add_argument("--num_imgs", default=1000000, type=int)
parser.add_argument("--no_multiprocessing", action="store_true", default=False)
args = parser.parse_args()

os.makedirs(args.fold_ABC, exist_ok=True)

A_list = [f for f in os.listdir(args.fold_A) if is_image_file(f)]
num_imgs = min(args.num_imgs, len(A_list))

pool = Pool() if not args.no_multiprocessing else None

for name in A_list[:num_imgs]:
    path_A = os.path.join(args.fold_A, name)
    path_B = os.path.join(args.fold_B, name)
    path_C = os.path.join(args.fold_C, name)

    if not os.path.isfile(path_B):
        print(f"[SKIP] Missing in B: {name}")
        continue
    if not os.path.isfile(path_C):
        print(f"[SKIP] Missing in C: {name}")
        continue

    out_name = f"{os.path.splitext(name)[0]}_concat_ABC.png"
    out_path = os.path.join(args.fold_ABC, out_name)

    if pool is not None:
        pool.apply_async(resize_and_concatenate_three, args=(path_A, path_B, path_C, out_path))
    else:
        resize_and_concatenate_three(path_A, path_B, path_C, out_path)

if pool is not None:
    pool.close()
    pool.join()

print("Done!")
