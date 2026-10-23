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
import json
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
PREVIEW_W, PREVIEW_H = 1100, 620        #图片分辨率大小

# mode → {真实参数名: (初始值, 最大值)}
PARAM_MAP = {
    "none":        {},
    "gray":        {},
    "gaussian":    {"ksize": (5, 31), "sigmaX": (0, 100)},  # ksize=核大小(自动奇数), sigmaX*0.1
    "median":      {"ksize": (5, 31)},                     # ksize=核大小(自动奇数)
    "bilateral":   {"d": (9, 30), "sigmaColor": (75, 200), "sigmaSpace": (75, 200)},
    "canny":       {"threshold1": (80, 255), "threshold2": (160, 500)},
    "adaptive":    {"blockSize": (5, 31), "C": (0, 20)},   # blockSize 自动奇数
    "binary":      {"thresh": (128, 255)},
    "threshold":   {"thresh": (128, 255)},
    "equalize":    {},
    "clahe":       {"clipLimit": (2, 10), "tileGridSize": (8, 32)},  # clipLimit*1.0, tileGridSize=块大小
    "brightness":  {"alpha": (10, 30), "beta": (0, 255)},   # alpha*0.1 即亮度系数 1.0~3.0; beta=0~255 加亮
    "saltpepper":  {"prob": (1, 50)},                       # prob/1000 即 0.1%~5%
    "rotate":      {"angle": (0, 360)},                     # 角度 0~360
    "resize":      {"scale": (5, 30)},                     # scale*0.1 即 0.5~3.0
    "crop":        {"ratio": (30, 100)},                    # ratio/100 即裁剪比例 0.3~1.0
    "flip":        {"flipCode": (0, 2)},                    # 0=水平, 1=垂直, 2=双向
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

def collect_params(mode):
    """从 PARAMS_WIN 读当前功能所有滑块的实时值，返回 {参数名: int}"""
    name = MODES[mode][0]
    return {pname: cv2.getTrackbarPos(pname, PARAMS_WIN) for pname in PARAM_MAP[name]}

def apply_filter(img, mode, params):
    """按 mode 调用 OpenCV 函数，params 是 collect_params 返回的 {参数名: int}"""
    name = MODES[mode][0]

    # —— 不需要任何参数的功能 ——
    if name == "none":
        return img.copy()
    if name == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if name == "equalize":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)

    # —— 核大小类滤波（ksize 必须为正奇数）——
    if name == "gaussian":
        # cv2.GaussianBlur(src, ksize, sigmaX[, sigmaY, borderType])
        # ksize=高斯核大小(正奇数, 越大越模糊); sigmaX=X方向标准差(0=由ksize自动算)
        k = params["ksize"]
        if k % 2 == 0: k += 1
        return cv2.GaussianBlur(img, (k, k), params["sigmaX"] / 10.0)
    if name == "median":
        # cv2.medianBlur(src, ksize)
        # ksize=中值滤波核大小(正奇数, 越大去噪越强也越模糊)
        k = params["ksize"]
        if k % 2 == 0: k += 1
        return cv2.medianBlur(img, k)
    if name == "bilateral":
        # cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace[, borderType])
        # d=邻域直径(≤0时由sigmaSpace算, 越大越慢);
        # sigmaColor=颜色空间标准差(越大颜色相差大的像素也被平滑);
        # sigmaSpace=坐标空间标准差(越大距离远的像素也被考虑, 与sigmaColor通常取相近值)
        d = max(1, params["d"])
        return cv2.bilateralFilter(img, d, params["sigmaColor"], params["sigmaSpace"])

    # —— 边缘 / 阈值类（多需先转灰度）——
    if name == "canny":
        # cv2.Canny(image, threshold1, threshold2[, edges, apertureSize, L2gradient])
        # threshold1=低阈值(弱边缘起点, 低于它的边缘被丢弃);
        # threshold2=高阈值(强边缘起点, 高于它的边缘被保留), 常用约为低阈值的 2~3 倍
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, params["threshold1"], params["threshold2"])
    if name == "adaptive":
        # cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
        # blockSize=计算局部均值的邻域大小(奇数且>1);
        # C=从均值里减去的常数(越大越暗)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        block = params["blockSize"]
        if block % 2 == 0: block += 1
        if block < 3: block = 3
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, block, params["C"])
    if name in ("binary", "threshold"):
        # cv2.threshold(src, thresh, maxval, type[, dst])
        # thresh=阈值(像素>thresh→maxval, 否则→0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, result = cv2.threshold(gray, params["thresh"], 255, cv2.THRESH_BINARY)
        return result

    # —— 局部均衡 / 亮度 ——
    if name == "clahe":
        # cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))
        # clipLimit=对比度限幅(越大对比越强, 但噪声也越明显);
        # tileGridSize=分块大小(把图切成 tile×tile 分别做局部均衡, 越大越接近全局均衡)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tile = params["tileGridSize"]
        clahe = cv2.createCLAHE(clipLimit=float(params["clipLimit"]), tileGridSize=(tile, tile))
        return clahe.apply(gray)
    if name == "brightness":
        # cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
        # alpha=对比度增益(实际 alpha*0.1 → 1.0~3.0, 越大对比越强);
        # beta=亮度偏移(0~255, 正值加亮负值变暗)
        return cv2.convertScaleAbs(img, alpha=params["alpha"] / 10.0, beta=params["beta"])

    # —— 噪声 / 几何变换 ——
    if name == "saltpepper":
        # prob=噪声比例(实际 prob/1000 → 0.1%~5%), 黑白点各占一半
        prob = params["prob"] / 1000.0
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
        # cv2.getRotationMatrix2D(center, angle, scale) + cv2.warpAffine(src, M, dsize)
        # angle=绕图像中心旋转的角度(正值逆时针, 负值顺时针)
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, params["angle"], 1.0)
        return cv2.warpAffine(img, M, (w, h))
    if name == "resize":
        # cv2.resize(src, dsize[, fx, fy, interpolation])
        # fx/fy=缩放比例(实际 scale*0.1 → 0.5~3.0, >1放大 <1缩小)
        scale = params["scale"] * 0.1
        h, w = img.shape[:2]
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if name == "crop":
        # 从图像中心按比例裁剪，ratio=保留比例(实际 ratio/100 → 0.3~1.0)
        ratio = params["ratio"] / 100.0
        h, w = img.shape[:2]
        ch, cw = int(h * ratio), int(w * ratio)
        y1, x1 = (h - ch) // 2, (w - cw) // 2
        return img[y1:y1 + ch, x1:x1 + cw]
    if name == "flip":
        # cv2.flip(src, flipCode)
        # 官方: flipCode=0 绕x轴(上下), 1 绕y轴(左右), -1 双向;
        # 滑块 0/1/2 通过 dir_map 映射为 1/0/-1 → 左右/上下/双向
        dir_map = {0: 1, 1: 0, 2: -1}
        code = dir_map.get(params["flipCode"], 1)
        return cv2.flip(img, code)

    return img.copy()

def save_result(src, result, mode, out_dir):
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{src.stem}_{MODES[mode][0]}.png"
    cv2.imwrite(str(path), result)
    return path


def batch(files, mode, params, out_dir):
    out_dir.mkdir(exist_ok=True)
    for i, src in enumerate(files, 1):
        img = cv2.imread(str(src))
        if img is None:
            print(f"跳过无法读取的图片: {src}")
            continue
        result = apply_filter(img, mode, params)
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
    def save_config(folder,mode,params,t):
        cfg = {"mode": MODES[mode][0],"params":params}
        json_path = os.path.join(os.path.dirname(folder),"config",f"{t}——{MODES[cur_mode][0]}.json")
        with open(json_path,"w",encoding="utf-8")as f:
            json.dump(cfg,f)

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

            # 从 PARAMS_WIN 读取当前功能的所有参数
            params = collect_params(cur_mode)
            result = apply_filter(cache["image"], cur_mode, params)
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
            params = collect_params(cur_mode)
            save_config(folder,cur_mode,params,"s")
            result = apply_filter(cache["image"], cur_mode, params)
            path = save_result(files[index], result, cur_mode, out_dir)
            print(f"已保存: {path}")
        elif key == ord("b"):
            cur_mode = cv2.getTrackbarPos("filter", WINDOW)
            params = collect_params(cur_mode)
            save_config(folder,cur_mode,params,"b")
            batch(files, cur_mode, params, out_dir)
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
