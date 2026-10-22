# 图像滤波工具箱

一个单文件的 OpenCV 图像滤波工具：选择文件夹，拖动滑杆实时预览，保存当前结果，或一键批量处理。



## 功能

- 递归读取文件夹中的 `.jpg`、`.jpeg`、`.png`、`.bmp`
- 6 种模式：原图、灰度、高斯滤波、中值滤波、双边滤波、Canny 边缘
- 5 条 Trackbar 实时调节参数
- 左侧原图、右侧处理结果对比
- `s` 保存当前图，`b` 批量处理整个文件夹
- 结果统一输出到 `output_simple/`

## 目录结构

```text
image_toolbox_init/
├── simple_gui.py     # 全部代码
├── README.md
├── samples/          # 测试图片
│   ├── shapes.png
│   ├── texture.jpg
│   ├── lowcontrast.png
│   └── sub/grid.bmp
└── output_simple/    # 保存/批量输出目录
```

## 环境要求

- Python 3.10+
- OpenCV
- NumPy
- Tkinter（只用于启动时选择文件夹）

```bash
python3 -m pip install opencv-python numpy
```

Linux 如果提示找不到 tkinter：

```bash
sudo apt install python3-tk
```

Windows / macOS 的官方 Python 安装包通常自带 Tkinter。

## 运行

```bash
cd image_toolbox_init
python3 simple_gui.py
```

启动后会弹出文件夹选择框：

- 选择包含图片的文件夹即可开始
- 直接取消则使用自带的 `samples/`

不需要任何命令行参数。

## 界面参数

窗口下方有 5 条 Trackbar，名称固定，当前模式只使用其中一部分：

| Trackbar | 范围 | 默认 | 作用 |
| --- | --- | --- | --- |
| `filter` | 0..5 | 0 | 切换滤波模式 |
| `size` | 1..31 | 5 | 高斯/中值的核大小；偶数会自动加 1 变成奇数 |
| `sigma` | 0..10 | 0 | 高斯标准差；双边滤波中映射为 `max(1, sigma × 15)` |
| `d` | 1..15 | 5 | 双边滤波的邻域直径 |
| `canny low` | 0..255 | 80 | Canny 低阈值，高阈值自动取 `max(1, min(255, low × 2))` |

说明：

- `size` 只影响高斯和中值；
- `sigma` 只影响高斯和双边；
- `d` 只影响双边；
- `canny low` 只影响 Canny；
- 当前模式用不到的滑杆会被忽略，不会报错；
- 窗口标题会显示当前模式、参数值和图片序号。

## 滤波模式

| 序号 | 模式 | 使用的参数 | 说明 |
| --- | --- | --- | --- |
| 0 | 原图 | 无 | 不做处理，方便对比 |
| 1 | 灰度 | 无 | BGR 转单通道灰度 |
| 2 | 高斯滤波 | `size`、`sigma` | 适合一般降噪/模糊 |
| 3 | 中值滤波 | `size` | 适合去除椒盐噪声，`3..7` 常用 |
| 4 | 双边滤波 | `d`、`sigma` | 去噪同时尽量保留边缘 |
| 5 | Canny 边缘 | `canny low` | 输出单通道黑白边缘图 |

## 快捷键

| 按键 | 功能 |
| --- | --- |
| `n` | 下一张 |
| `p` | 上一张 |
| `s` | 保存当前处理结果 |
| `b` | 批量处理当前文件夹全部图片 |
| `o` | 重新选择文件夹 |
| `q` / `Esc` | 退出 |

## 输出规则

- 输出目录固定为运行目录下的 `output_simple/`；
- 文件名格式：`原文件名_滤波名.png`，例如 `shapes_gaussian.png`；
- 批量处理会递归处理当前文件夹下的所有图片，输出保持扁平结构，不保留子目录；
- 同名结果会直接覆盖；
- 所有结果统一存为 PNG；
- 原始图片不会被修改。

## 说明

- 一次只应用一种滤波，不支持多步串联；
- 批量处理是顺序执行的，处理期间窗口会短暂卡住，终端会打印进度；
- 直接使用 `cv2.imread` / `cv2.imwrite`，没有做特殊路径兼容处理；如果遇到读取失败，终端会打印提示并跳过该文件；
