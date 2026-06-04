# virtual-try-on

基于 CP-VTON 的 2D 虚拟试衣 demo。输入人物照片和平铺服装图，输出试穿合成图。

## 做了什么

在 OpenCV 官方 CP-VTON 示例和 `virtual_try_on_use_deep_learning` 开源管线基础上，补了关键点驱动的几何对齐，让不同衣长、版型的上装也能跑通。

主要改动：

- 设计了 12 关键点标注（领口、肩、袖、腋下、下摆等），整理了约 100 组人-衣配对，存在 `data/final_combined_pairs.json`
- 写了 `cloth_alignment.py`：躯干平移 + 下摆透视变形，替代/补充 GMM 形变
- 在 `full_geometry.py` 里扩展 Agnostic 下摆区域，并对 TOM 合成掩码做了局部增强
- 保留了三个版本的入口，方便对比：`main.py`（GMM 基线）→ `version2.0.py`（TPS）→ `full_geometry.py`（几何对齐，推荐）

## 目录结构

```
virtual-try-on/
├── src/                 # 源码
├── models/              # 模型权重
├── data/                # 配对数据与样本图
│   ├── final_combined_pairs.json
│   └── raw_images/
├── samples/             # 内置测试图
│   ├── test_img/
│   └── test_color/
├── requirements.txt
└── README.md
```

## 环境

- Python 3.8+
- opencv-python
- numpy

```bash
pip install -r requirements.txt
```

## 运行

在项目根目录执行：

```bash
# 推荐：完整几何对齐方案
python src/full_geometry.py

# 基线：GMM + TOM
python src/main.py

# 迭代版：JSON 关键点 + TPS
python src/version2.0.py
```

指定输入：

```bash
python src/full_geometry.py --input_image samples/test_img/000074_0.jpg --input_cloth samples/test_color/000048_1.jpg
```

默认会读取 `data/final_combined_pairs.json` 的第一组配对来跑 demo。

## 模型文件说明

`models/` 目录下共 5 个权重相关文件，来源如下：

| 文件 | 说明 |
|------|------|
| `cp_vton_gmm.onnx` | 第三方预训练，几何匹配模块（GMM） |
| `lip_jppnet_384.pb` | 第三方预训练，人体解析（JPPNet） |
| `openpose_pose_coco.prototxt` | 第三方，OpenPose 网络结构 |
| `openpose_pose_coco.caffemodel` | 第三方预训练，OpenPose 姿态估计 |
| `cp_vton_tom.onnx` | **自行训练/导出的试穿融合模块（TOM）** |

前四个文件来自上游开源工程/OpenCV 示例配套，不是本项目训练出来的，具体发布来源没逐一追溯。最终方案里 GMM 可被几何对齐替代，TOM 仍负责试穿融合。

> 大体积权重通过 Git LFS 管理，clone 后若模型缺失，先执行 `git lfs pull`。

## 数据格式

`final_combined_pairs.json` 每条记录包含：

- `pair`：配对名，如 `01_c_07_h`
- `cloth_points`：平铺服装图上的 12 个点
- `human_points`：人物原穿着上的参考点
- `pair_points`：本次试穿的目标位置

对应图片放在 `data/raw_images/clothes/` 和 `data/raw_images/humans/`。

## 参考

- CP-VTON 论文与 OpenCV `samples/dnn/virtual_try_on.py`
- 上游参考仓库：`virtual_try_on_use_deep_learning`
