"""极简版图像滤波工具：一个窗口，常用滤波，实时预览。

运行：python3 simple_gui.py

按键：
    n / p   下一张 / 上一张
    s       保存当前结果
    b       批量处理当前文件夹
    o       重新选择文件夹
    q / Esc 退出
"""

import os
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import numpy as np

WINDOW = "Image Filters"#窗口名字
MODES = [
    ("none", "原图"),
    ("gray", "灰度"),
    ("gaussian", "高斯滤波"),
    ("median", "中值滤波"),
    ("bilateral", "双边滤波"),
    ("canny", "Canny 边缘"),
]#功能列表
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}#支持的文件类型
OUTPUT = Path("output_simple")#图片输出的位置
PREVIEW_W, PREVIEW_H = 1100, 620#图片分辨率大小


def choose_folder():
    root = Tk()#创建一个TK窗口
    # root.withdraw()#把它隐藏
    folder = filedialog.askdirectory(title="选择图片文件夹")#弹出系统文件夹选择框
    root.destroy()#用完销毁
    return folder# 返回文件夹内容


def list_images(folder):
    files = []
    for root, _dirs, names in os.walk(folder):#walk拿到文件夹路径循环拿到文件夹里的图片
        for name in names:
            if Path(name).suffix.lower() in EXTS:#取文件扩展名，然后转换成小写，判断是否在文件类型里
                files.append(Path(root) / name)#将文件名拼接到一起，形成路径
    return sorted(files)#返回排列好的文件


def to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def fit(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def preview(original, result):
    left = fit(original, PREVIEW_W // 2 - 8, PREVIEW_H)
    right = fit(to_bgr(result), PREVIEW_W // 2 - 8, PREVIEW_H)
    height = max(left.shape[0], right.shape[0])
    left = cv2.copyMakeBorder(left, 0, height - left.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    right = cv2.copyMakeBorder(right, 0, height - right.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    gap = np.full((height, 16, 3), (30, 30, 30), np.uint8)
    return np.hstack([left, gap, right])


def apply_filter(img, mode, size, sigma, d, low):
    name = MODES[mode][0]

    if name == "none":
        return img.copy()
    if name == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    k = max(1, size)
    if k % 2 == 0:
        k += 1

    if name == "gaussian":
        return cv2.GaussianBlur(img, (k, k), sigma)
    if name == "median":
        return cv2.medianBlur(img, k)
    if name == "bilateral":
        return cv2.bilateralFilter(img, max(1, d), max(1, sigma * 15), max(1, sigma * 15))
    if name == "canny":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, low, max(1, min(255, low * 2)))
    return img.copy()


def save_result(src, result, mode):
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / f"{src.stem}_{MODES[mode][0]}.png"
    cv2.imwrite(str(path), result)
    return path


def batch(files, mode, size, sigma, d, low):
    OUTPUT.mkdir(exist_ok=True)
    for i, src in enumerate(files, 1):
        img = cv2.imread(str(src))
        if img is None:
            print(f"跳过无法读取的图片: {src}")
            continue
        result = apply_filter(img, mode, size, sigma, d, low)
        path = OUTPUT / f"{src.stem}_{MODES[mode][0]}.png"
        cv2.imwrite(str(path), result)
        print(f"[{i}/{len(files)}] {src.name} -> {path.name}")


def main():
    folder = choose_folder()
    if not folder:
        folder = "samples"
    files = list_images(folder)
    if not files:
        print(f"没有找到图片: {folder}")
        return

    index = 0
    cache = {"index": -1, "image": None}
    dirty = True

    def changed(_value):
        nonlocal dirty#通过这个关键字修改dirty值
        dirty = True

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)#创建一个窗口
    cv2.createTrackbar("filter", WINDOW, 0, len(MODES) - 1, changed)#创建一个滚动条，参数分别为滚动条名称，为滚动条所依附窗口名，初始值，最大值，回调函数
    cv2.createTrackbar("size", WINDOW, 5, 31, changed)
    cv2.createTrackbar("sigma", WINDOW, 0, 10, changed)
    cv2.createTrackbar("d", WINDOW, 5, 15, changed)
    cv2.createTrackbar("canny low", WINDOW, 80, 255, changed)

    print("n/p 切图  s 保存  b 批量  o 换文件夹  q 退出")

    while True:
        if dirty:
            if cache["index"] != index:
                img = cv2.imread(str(files[index]))
                if img is None:
                    print(f"无法读取: {files[index]}")
                    return
                cache["index"] = index
                cache["image"] = img

            mode = cv2.getTrackbarPos("filter", WINDOW)
            size = cv2.getTrackbarPos("size", WINDOW)
            sigma = cv2.getTrackbarPos("sigma", WINDOW)
            d = cv2.getTrackbarPos("d", WINDOW)
            low = cv2.getTrackbarPos("canny low", WINDOW)

            result = apply_filter(cache["image"], mode, size, sigma, d, low)
            cv2.imshow(WINDOW, preview(cache["image"], result))
            cv2.setWindowTitle(
                WINDOW,
                f"{MODES[mode][0]} | size={size} sigma={sigma} d={d} "
                f"canny={low} | {index + 1}/{len(files)} {files[index].name}",
            )
            dirty = False

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("n"):
            index = (index + 1) % len(files)
            dirty = True
        elif key == ord("p"):
            index = (index - 1) % len(files)
            dirty = True
        elif key == ord("s"):
            mode = cv2.getTrackbarPos("filter", WINDOW)
            size = cv2.getTrackbarPos("size", WINDOW)
            sigma = cv2.getTrackbarPos("sigma", WINDOW)
            d = cv2.getTrackbarPos("d", WINDOW)
            low = cv2.getTrackbarPos("canny low", WINDOW)
            result = apply_filter(cache["image"], mode, size, sigma, d, low)
            path = save_result(files[index], result, mode)
            print(f"已保存: {path}")
        elif key == ord("b"):
            mode = cv2.getTrackbarPos("filter", WINDOW)
            size = cv2.getTrackbarPos("size", WINDOW)
            sigma = cv2.getTrackbarPos("sigma", WINDOW)
            d = cv2.getTrackbarPos("d", WINDOW)
            low = cv2.getTrackbarPos("canny low", WINDOW)
            batch(files, mode, size, sigma, d, low)
        elif key == ord("o"):
            folder = choose_folder()
            if folder:
                new_files = list_images(folder)
                if new_files:
                    files = new_files
                    index = 0
                    cache["index"] = -1
                    dirty = True
                else:
                    print(f"这个文件夹里没有图片: {folder}")
            print(f"当前文件夹: {folder}，共 {len(files)} 张")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
