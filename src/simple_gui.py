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
    ("adaptive", "自适应阈值"),
    ("binary", "二值化"),
    ("threshold", "固定阈值二值化"),
    ("equalize", "直方图均衡化"),
    ("clahe", "CLAHE局部均衡"),
    ("brightness", "亮度/对比度"),
    ("saltpepper", "椒盐噪声"),
    ("rotate", "旋转"),
    ("resize", "等比缩放"),
    ("crop", "中心裁剪"),
    ("flip", "镜像翻转"),
]#功能列表
PARAMS_WIN = "Params"        # 参数窗口
PREVIEW_WIN = WINDOW         # 预览窗口
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}#支持的文件类型
PREVIEW_W, PREVIEW_H = 1100, 620#图片分辨率大小
# mode → {参数名: (初始值, 最大值)}
PARAM_MAP = {
    "none":        {},
    "gray":        {},
    "gaussian":    {"size": (5, 31), "sigma": (0, 10)},
    "median":      {"size": (5, 31)},
    "bilateral":   {"size": (5, 31), "sigma": (0, 10), "d": (5, 15)},
    "canny":       {"canny low": (80, 255)},
    "adaptive":    {"d": (5, 15), "canny low": (128, 255)},
    "binary":      {"canny low": (128, 255)},
    "threshold":   {"canny low": (128, 255)},
    "equalize":    {},
    "clahe":       {"d": (8, 32)},
    "brightness":  {"sigma": (5, 10), "d": (5, 15)},
    "saltpepper":  {"sigma": (0, 10)},
    "rotate":      {"sigma": (0, 10)},
    "resize":      {"sigma": (5, 10)},
    "crop":        {"size": (20, 31)},
    "flip":        {"sigma": (0, 2)},
}


def choose_folder(title="选择图片文件夹"):
    root = Tk()#创建一个TK窗口
    root.withdraw()#把它隐藏
    folder = filedialog.askdirectory(title=title)#弹出系统文件夹选择框
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
    if name == "adaptive":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        block = max(3, d)
        if block % 2 == 0:
            block += 1
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, block, low - 128)
    if name in ("binary", "threshold"):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, result = cv2.threshold(gray, low, 255, cv2.THRESH_BINARY)
        return result
    if name == "equalize":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)
    if name == "clahe":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(max(1, d), max(1, d)))
        return clahe.apply(gray)
    if name == "brightness":
        # sigma 0~10 映射亮度 +0.5~+1.5，d 映射对比度 0.5~2.0
        alpha = 0.5 + sigma / 10.0
        beta = int((d - 5) * 10)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    if name == "saltpepper":
        # sigma 0~10 映射噪声比例 0~5%
        prob = sigma / 200.0
        noisy = img.copy()
        rand = np.random.random(noisy.shape[:2])
        black = rand < prob / 2
        white = rand > 1 - prob / 2
        if noisy.ndim == 3:
            noisy[black] = [0, 0, 0]
            noisy[white] = [255, 255, 255]
        else:
            noisy[black] = 0
            noisy[white] = 255
        return noisy
    if name == "rotate":
        # sigma 0~10 映射角度 0~360 度
        angle = sigma * 36.0
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h))
    if name == "resize":
        # sigma 0~10 映射比例 0.5~2.0
        scale = 0.5 + sigma / 10.0
        h, w = img.shape[:2]
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if name == "crop":
        # size 映射裁剪比例 0.1~1.0
        ratio = 0.1 + k / 31.0 * 0.9
        h, w = img.shape[:2]
        ch, cw = int(h * ratio), int(w * ratio)
        y1, x1 = (h - ch) // 2, (w - cw) // 2
        return img[y1:y1 + ch, x1:x1 + cw]
    if name == "flip":
        # sigma 0/1/2 映射 水平/垂直/双向
        dir_map = {0: 1, 1: 0, 2: -1}
        code = dir_map.get(sigma % 3, 1)
        return cv2.flip(img, code)

    return img.copy()



def save_result(src, result, mode, out_dir):
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{src.stem}_{MODES[mode][0]}.png"
    cv2.imwrite(str(path), result)
    return path


def batch(files, mode, size, sigma, d, low, out_dir):
    out_dir.mkdir(exist_ok=True)
    for i, src in enumerate(files, 1):
        img = cv2.imread(str(src))
        if img is None:
            print(f"跳过无法读取的图片: {src}")
            continue
        result = apply_filter(img, mode, size, sigma, d, low)
        path = out_dir / f"{src.stem}_{MODES[mode][0]}.png"
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

    out_dir = Path(folder).parent / "output_sample"  # 默认输出目录：图片目录的同级

    index = 0
    cache = {"index": -1, "image": None}
    dirty = True

    def changed(_value):
        nonlocal dirty
        dirty = True

    # 预览窗口：只放 filter 选择
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("filter", WINDOW, 0, len(MODES) - 1, changed)

    # 参数窗口：当前功能需要的 trackbar
    cv2.namedWindow(PARAMS_WIN, cv2.WINDOW_AUTOSIZE)

    def rebuild_params_trackbars(mode):
        cv2.destroyWindow(PARAMS_WIN)
        cv2.namedWindow(PARAMS_WIN, cv2.WINDOW_AUTOSIZE)
        name = MODES[mode][0]
        for param, (val, mx) in PARAM_MAP[name].items():
            cv2.createTrackbar(param, PARAMS_WIN, val, mx, changed)

    rebuild_params_trackbars(0)  # 初始建一次

    print("n/p 切图  s 保存  b 批量  k 选输出  o 换文件夹  q 退出")
    print(f"当前输出目录: {out_dir}")

    prev_mode = 0  # 上一次的 filter，用来检测变化

    while True:
        # 检测 filter 变化 → 重建参数窗口
        cur_mode = cv2.getTrackbarPos("filter", WINDOW)
        if cur_mode != prev_mode:
            prev_mode = cur_mode
            rebuild_params_trackbars(cur_mode)
            dirty = True

        if dirty:
            if cache["index"] != index:
                img = cv2.imread(str(files[index]))
                if img is None:
                    print(f"无法读取: {files[index]}")
                    return
                cache["index"] = index
                cache["image"] = img

            # 从 PARAMS_WIN 读参数，不存在的给默认值
            cur_name = MODES[cur_mode][0]
            size = cv2.getTrackbarPos("size", PARAMS_WIN) if "size" in PARAM_MAP[cur_name] else 5
            sigma = cv2.getTrackbarPos("sigma", PARAMS_WIN) if "sigma" in PARAM_MAP[cur_name] else 0
            d = cv2.getTrackbarPos("d", PARAMS_WIN) if "d" in PARAM_MAP[cur_name] else 5
            low = cv2.getTrackbarPos("canny low", PARAMS_WIN) if "canny low" in PARAM_MAP[cur_name] else 80

            result = apply_filter(cache["image"], cur_mode, size, sigma, d, low)
            cv2.imshow(WINDOW, preview(cache["image"], result))
            cv2.setWindowTitle(
                WINDOW,
                f"{MODES[cur_mode][0]} | {index + 1}/{len(files)} {files[index].name}",
            )
            dirty = False

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("]"):
            index = (index + 1) % len(files)
            dirty = True
        elif key == ord("["):
            index = (index - 1) % len(files)
            dirty = True
        elif key == ord("s"):
            cur_mode = cv2.getTrackbarPos("filter", WINDOW)
            cur_name = MODES[cur_mode][0]
            size = cv2.getTrackbarPos("size", PARAMS_WIN) if "size" in PARAM_MAP[cur_name] else 5
            sigma = cv2.getTrackbarPos("sigma", PARAMS_WIN) if "sigma" in PARAM_MAP[cur_name] else 0
            d = cv2.getTrackbarPos("d", PARAMS_WIN) if "d" in PARAM_MAP[cur_name] else 5
            low = cv2.getTrackbarPos("canny low", PARAMS_WIN) if "canny low" in PARAM_MAP[cur_name] else 80
            result = apply_filter(cache["image"], cur_mode, size, sigma, d, low)
            path = save_result(files[index], result, cur_mode, out_dir)
            print(f"已保存: {path}")
        elif key == ord("b"):
            cur_mode = cv2.getTrackbarPos("filter", WINDOW)
            cur_name = MODES[cur_mode][0]
            size = cv2.getTrackbarPos("size", PARAMS_WIN) if "size" in PARAM_MAP[cur_name] else 5
            sigma = cv2.getTrackbarPos("sigma", PARAMS_WIN) if "sigma" in PARAM_MAP[cur_name] else 0
            d = cv2.getTrackbarPos("d", PARAMS_WIN) if "d" in PARAM_MAP[cur_name] else 5
            low = cv2.getTrackbarPos("canny low", PARAMS_WIN) if "canny low" in PARAM_MAP[cur_name] else 80
            batch(files, cur_mode, size, sigma, d, low, out_dir)
        elif key == ord("k"):
            new_out = choose_folder("选择输出目录")
            if new_out:
                out_dir = Path(new_out)
                print(f"输出目录已改为: {out_dir}")
            else:
                print("未选择，保持不变")
        elif key == ord("o"):
            folder = choose_folder()
            if folder:
                new_files = list_images(folder)
                if new_files:
                    files = new_files
                    index = 0
                    cache["index"] = -1
                    out_dir = Path(folder).parent / "output_sample"  # 换文件夹时重置输出目录
                    print(f"输出目录重置为: {out_dir}")
                    dirty = True
                else:
                    print(f"这个文件夹里没有图片: {folder}")
            print(f"当前文件夹: {folder}，共 {len(files)} 张")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
