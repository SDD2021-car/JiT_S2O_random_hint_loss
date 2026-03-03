import os
import numpy as np
import cv2
import argparse
from multiprocessing import Pool
import re


def extract_key(filename):
    """
    从文件名中提取 ROI + season + index + patch。
    返回类似：ROIs1158_spring_18_p642
    """
    pattern = re.compile(r"(ROIs\d+_[a-zA-Z]+)_s\d+_(\d+)_p(\d+)")
    m = pattern.search(filename)
    if m:
        return f"{m.group(1)}_{m.group(2)}_p{m.group(3)}"
    return None


def resize_and_concatenate_three(path_A, path_B, path_C, path_ABC):
    im_A = cv2.imread(path_A, 1)
    im_B = cv2.imread(path_B, 1)
    im_C = cv2.imread(path_C, 1)

    if im_A is None or im_B is None or im_C is None:
        print(f"[WARN] Read failed: A={path_A}, B={path_B}, C={path_C}")
        return

    hA, wA = im_A.shape[:2]
    hB, wB = im_B.shape[:2]
    hC, wC = im_C.shape[:2]

    target_h = min(hA, hB, hC)
    target_w = min(wA, wB, wC)

    im_A_r = cv2.resize(im_A, (target_w, target_h))
    im_B_r = cv2.resize(im_B, (target_w, target_h))
    im_C_r = cv2.resize(im_C, (target_w, target_h))

    im_ABC = np.concatenate([im_A_r, im_B_r, im_C_r], axis=1)
    cv2.imwrite(path_ABC, im_ABC)


# -------------------- parser --------------------
parser = argparse.ArgumentParser('create image triplets')
parser.add_argument('--fold_A', default="/NAS_data/yjy/ColorS2O_random_hint_concat/hint_output/dot/hints/overlay", type=str)
parser.add_argument('--fold_B', default="/NAS_data/yjy/ColorS2O_random_hint_concat/hint_output/dot/hints/color", type=str)
parser.add_argument('--fold_C', default="/NAS_data/hjf/JiTcolor/outputs/SAR2Opt/caJiT_CP/round4/noLoss_noHintsDropout_dot_concat/heun-steps50-cfg1.0-interval0.0-1.0-image50000-res512", type=str)
parser.add_argument('--fold_ABC', default="/NAS_data/yjy/ColorS2O_random_hint_concat/combine", type=str)
parser.add_argument('--num_imgs', type=int, default=1000000)
parser.add_argument('--no_multiprocessing', action='store_true', default=False)
args = parser.parse_args()

img_fold_A = args.fold_A
img_fold_B = args.fold_B
img_fold_C = args.fold_C
img_fold_ABC = args.fold_ABC

os.makedirs(img_fold_ABC, exist_ok=True)

# -------------------- build B/C dictionaries --------------------
def build_key_dict(folder):
    d = {}
    for fname in os.listdir(folder):
        key = extract_key(fname)
        if key:
            d[key] = fname
    return d

B_dict = build_key_dict(img_fold_B)
C_dict = build_key_dict(img_fold_C)

print(f"Found {len(B_dict)} keys in folder B.")
print(f"Found {len(C_dict)} keys in folder C.")

# -------------------- process A --------------------
A_list = os.listdir(img_fold_A)
num_imgs = min(args.num_imgs, len(A_list))

pool = Pool() if not args.no_multiprocessing else None

for idx, name_A in enumerate(A_list[:num_imgs]):
    key_A = extract_key(name_A)
    if key_A is None:
        print(f"Skip (cannot extract key): {name_A}")
        continue

    if key_A not in B_dict:
        print(f"No match in B for: {name_A}")
        continue

    if key_A not in C_dict:
        print(f"No match in C for: {name_A}")
        continue

    name_B = B_dict[key_A]
    name_C = C_dict[key_A]

    path_A = os.path.join(img_fold_A, name_A)
    path_B = os.path.join(img_fold_B, name_B)
    path_C = os.path.join(img_fold_C, name_C)

    # 输出名：保留A名，追加 concat_ABC
    out_name = f"{os.path.splitext(name_A)[0]}_concat_ABC.png"
    path_ABC = os.path.join(img_fold_ABC, out_name)

    if not args.no_multiprocessing:
        pool.apply_async(resize_and_concatenate_three, args=(path_A, path_B, path_C, path_ABC))
    else:
        resize_and_concatenate_three(path_A, path_B, path_C, path_ABC)

if not args.no_multiprocessing:
    pool.close()
    pool.join()

print("Done!")
