# import os
# from PIL import Image
#
# # 设置文件夹路径
# folder_path = '/data/yjy_data/SAM2/hint_outputs_train/color_hint_by_dots'  # 替换为你的文件夹路径
#
# # 遍历文件夹中的文件
# for filename in os.listdir(folder_path):
#     if filename.endswith('.png'):  # 检查是否为JPG或JPEG文件
#         # # 构建文件路径
#         img_path = os.path.join(folder_path, filename)
#         # os.remove(img_path)
#
#         # 打开JPG文件
#         img = Image.open(img_path)
#
#         # 获取文件名（不包括扩展名）
#         name_without_extension = os.path.splitext(filename)[0]
#
#         # 构建PNG文件保存路径
#         png_path = os.path.join(folder_path, f"{name_without_extension}.jpg")
#
#         # 保存为PNG格式
#         img.save(png_path, 'JPG')
#
#         print(f"已将 {filename} 转换为 {png_path}")

import os
from PIL import Image

folder_path = '/data/yjy_data/SAM2/hint_outputs_test/color_hint_by_dots'

for filename in os.listdir(folder_path):
    if filename.lower().endswith('.png'):
        img_path = os.path.join(folder_path, filename)
        os.remove(img_path)
        # with Image.open(img_path) as img:
        #     # PNG 常见 RGBA / P，保存 JPG 前转 RGB
        #     if img.mode in ('RGBA', 'LA', 'P'):
        #         img = img.convert('RGB')
        #
        #     name_without_extension = os.path.splitext(filename)[0]
        #     jpg_path = os.path.join(folder_path, f"{name_without_extension}.jpg")
        #
        #     # 可选：避免覆盖
        #     if os.path.exists(jpg_path):
        #         print(f"跳过，已存在: {jpg_path}")
        #         continue
        #
        #     img.save(jpg_path, format='JPEG', quality=95, subsampling=0, optimize=True)
        #
        # print(f"已将 {filename} 转换为 {jpg_path}")
