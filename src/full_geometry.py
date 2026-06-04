import argparse
import os.path
import numpy as np
import cv2 as cv
import json
from common import findFile, PROJECT_ROOT, resolve_path
from human_parsing import parse_human
from cloth_alignment import align_and_warp_cloth
backends = (
    cv.dnn.DNN_BACKEND_DEFAULT,
    cv.dnn.DNN_BACKEND_HALIDE,
    cv.dnn.DNN_BACKEND_INFERENCE_ENGINE,
    cv.dnn.DNN_BACKEND_OPENCV
)
targets = (
    cv.dnn.DNN_TARGET_CPU,
    cv.dnn.DNN_TARGET_OPENCL,
    cv.dnn.DNN_TARGET_OPENCL_FP16,
    cv.dnn.DNN_TARGET_MYRIAD,
    cv.dnn.DNN_TARGET_HDDL
)

parser = argparse.ArgumentParser(
    description='Use this script to run virtual try-on using CP-VTON',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--input_image', type=str, default='samples/test_img/000074_0.jpg', help='Path to image with person.')
parser.add_argument('--input_cloth', type=str, default='samples/test_color/000048_1.jpg', help='Path to target cloth image')
parser.add_argument('--gmm_model', '-gmm', default='models/cp_vton_gmm.onnx', help='Path to Geometric Matching Module .onnx model.')
parser.add_argument('--tom_model', '-tom', default='models/cp_vton_tom.onnx', help='Path to Try-On Module .onnx model.')
parser.add_argument('--segmentation_model', default='models/lip_jppnet_384.pb', help='Path to cloth segmentation .pb model.')
parser.add_argument('--openpose_proto', default='models/openpose_pose_coco.prototxt', help='Path to OpenPose .prototxt model trained on COCO.')
parser.add_argument('--openpose_model', default='models/openpose_pose_coco.caffemodel', help='Path to OpenPose .caffemodel model trained on COCO.')
parser.add_argument(
    '--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
    help=("Choose one of computation backends: "
          "%d: automatically (by default), "
          "%d: Halide language, "
          "%d: Intel OpenVINO, "
          "%d: OpenCV implementation" % backends)
)
parser.add_argument(
    '--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
    help=('Choose one of target computation devices: '
          '%d: CPU (by default), '
          '%d: OpenCL, '
          '%d: OpenCL fp16, '
          '%d: NCS2 VPU, '
          '%d: HDDL VPU' % targets)
)
args, _ = parser.parse_known_args()


def get_pose_map(image, proto_path, model_path, backend, target, height=256, width=192):
    radius = 5
    inp = cv.dnn.blobFromImage(image, 1.0 / 255, (width, height))

    net = cv.dnn.readNet(proto_path, model_path)
    net.setPreferableBackend(backend)
    net.setPreferableTarget(target)
    net.setInput(inp)
    out = net.forward()

    threshold = 0.1
    _, out_c, out_h, out_w = out.shape
    pose_map = np.zeros((height, width, out_c - 1))
    # last label: Background
    for i in range(0, out.shape[1] - 1):
        heatMap = out[0, i, :, :]
        keypoint = np.full((height, width), -1)
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = width * point[0] // out_w
        y = height * point[1] // out_h
        if conf > threshold and x > 0 and y > 0:
            keypoint[y - radius:y + radius, x - radius:x + radius] = 1
        pose_map[:, :, i] = keypoint
    pose_map = pose_map.transpose(2, 0, 1)
    return pose_map


class BilinearFilter(object):
    """
    PIL bilinear resize implementation
    image = image.resize((image_width // 16, image_height // 16), Image.BILINEAR)
    """
    def _precompute_coeffs(self, inSize, outSize):
        filterscale = max(1.0, inSize / outSize)
        ksize = int(np.ceil(filterscale)) * 2 + 1

        kk = np.zeros(shape=(outSize * ksize,), dtype=np.float32)
        bounds = np.empty(shape=(outSize * 2,), dtype=np.int32)

        centers = (np.arange(outSize) + 0.5) * filterscale + 0.5
        bounds[::2] = np.where(centers - filterscale < 0, 0, centers - filterscale)
        bounds[1::2] = (
            np.where(centers + filterscale > inSize, inSize, centers + filterscale)
            - bounds[::2]
        )
        xmins = bounds[::2] - centers + 1

        points = np.array(
            [np.arange(row) + xmins[i] for i, row in enumerate(bounds[1::2])],
            dtype=object
        ) / filterscale
        for xx in range(0, outSize):
            point = points[xx]
            bilinear = np.where(point < 1.0, 1.0 - abs(point), 0.0)
            ww = np.sum(bilinear)
            kk[xx * ksize: xx * ksize + bilinear.size] = np.where(
                ww == 0.0, bilinear, bilinear / ww
            )
        return bounds, kk, ksize

    def _resample_horizontal(self, out, img, ksize, bounds, kk):
        for yy in range(0, out.shape[0]):
            for xx in range(0, out.shape[1]):
                xmin = bounds[xx * 2 + 0]
                xmax = bounds[xx * 2 + 1]
                k = kk[xx * ksize: xx * ksize + xmax]
                out[yy, xx] = np.round(np.sum(img[yy, xmin: xmin + xmax] * k))

    def _resample_vertical(self, out, img, ksize, bounds, kk):
        for yy in range(0, out.shape[0]):
            ymin = bounds[yy * 2 + 0]
            ymax = bounds[yy * 2 + 1]
            k = kk[yy * ksize: yy * ksize + ymax]
            out[yy] = np.round(
                np.sum(img[ymin: ymin + ymax, 0:out.shape[1]] * k[:, np.newaxis], axis=0)
            )

    def imaging_resample(self, img, xsize, ysize):
        height, width = img.shape[0:2]
        bounds_horiz, kk_horiz, ksize_horiz = self._precompute_coeffs(width, xsize)
        bounds_vert, kk_vert, ksize_vert = self._precompute_coeffs(height, ysize)

        out_hor = np.empty((img.shape[0], xsize), dtype=np.uint8)
        self._resample_horizontal(out_hor, img, ksize_horiz, bounds_horiz, kk_horiz)
        out = np.empty((ysize, xsize), dtype=np.uint8)
        self._resample_vertical(out, out_hor, ksize_vert, bounds_vert, kk_vert)
        return out


class CpVton(object):
    @staticmethod
    def _line_from_points(p1, p2):
        """
        给定两个点 p1, p2 = (x, y)，返回直线的一般式系数 (A, B, C):
            A x + B y + C = 0
        """
        x1, y1 = p1
        x2, y2 = p2
        # 垂直线和普通情况统一处理
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        return A, B, C

    @staticmethod
    def _line_intersection(l1, l2):
        """
        计算两条直线的交点。
        l1, l2 形式为 (A, B, C) 表示 A x + B y + C = 0
        返回 (x, y) 或 None（平行/重合时）
        """
        A1, B1, C1 = l1
        A2, B2, C2 = l2
        D = A1 * B2 - A2 * B1
        if abs(D) < 1e-8:
            return None  # 平行或几乎平行
        x = (B1 * C2 - B2 * C1) / D
        y = (C1 * A2 - C2 * A1) / D
        return (x, y)

    @staticmethod
    def _point_side_of_line(line, point):

        A, B, C = line
        x, y = point
        return A * x + B * y + C

    def extend_agnostic_with_hem(self,
                                 agnostic,
                                 human_points_scaled,
                                 pair_points_scaled,
                                 height=256,
                                 width=192):
        """
        使用 human_points_scaled / pair_points_scaled 对 agnostic 做二次处理：
        - 在“原下摆线 l1 与新衣下摆围成的区域”内，修改 res_shape (第0通道)，
          让这块区域的形状提示更像“有人体/衣服”，而不是背景。
        - 只改第0通道，其它通道保持不变。
        """
        agnostic_extended = agnostic.copy()  # (1, C, H, W)

        # 1. 构造下摆扩展 mask (H,W)
        hem_mask = self.build_extended_hem_mask(
            human_points_scaled,
            pair_points_scaled,
            height=height,
            width=width
        ).astype(np.uint8)  # 0/1

        if hem_mask.sum() == 0:
            # 没有需要扩展的区域
            return agnostic_extended

        hem_mask_4d = hem_mask[np.newaxis, np.newaxis, :, :].astype(np.float32)

        # 2. res_shape 在第0通道: shape (1,1,H,W)
        res_shape = agnostic_extended[:, 0:1, :, :]  # (1,1,H,W)

        # 3. 估计“人体区域”的典型值
        #    从现有非零区域里取一个中位数作为填充值
        res_np = res_shape[0, 0]  # (H,W)
        nonzero_vals = res_np[res_np > 0]
        if len(nonzero_vals) > 0:
            base = float(np.median(nonzero_vals))
            # 稍微抬高一点点，比如 +0.2，避免和背景太像
            fill_value = base + 0.2
        else:
            fill_value = 0.5  # 温和一点的兜底值
        # 4. 在 hem_mask==1 的区域，把 res_shape 填成 fill_value
        res_shape = res_shape * (1.0 - hem_mask_4d) + fill_value * hem_mask_4d
        agnostic_extended[:, 0:1, :, :] = res_shape

        return agnostic_extended
    def build_extended_hem_mask(self,
                                human_points_scaled,
                                pair_points_scaled,
                                height=256,
                                width=192):
        mask = np.zeros((height, width), dtype=np.uint8)

        if (human_points_scaled is None or len(human_points_scaled) < 8 or
                pair_points_scaled is None or len(pair_points_scaled) < 8):
            return mask

        # 取关键点（Python 下标从0开始）
        # human_points: 第6、第7个点 -> index 5, 6
        h6 = np.array(human_points_scaled[5], dtype=np.float32)
        h7 = np.array(human_points_scaled[6], dtype=np.float32)
        # pair_points: 第5、第6个点 -> index 4, 5
        p5 = np.array(pair_points_scaled[4], dtype=np.float32)
        p6 = np.array(pair_points_scaled[5], dtype=np.float32)
        # pair_points: 第7、第8个点 -> index 6, 7
        p7 = np.array(pair_points_scaled[6], dtype=np.float32)
        p8 = np.array(pair_points_scaled[7], dtype=np.float32)

        # 构造三条直线
        l1 = self._line_from_points(h6, h7)  # 原衣服下边缘
        l2 = self._line_from_points(p5, p6)  # 新衣左下摆边
        l3 = self._line_from_points(p7, p8)  # 新衣右下摆边

        # 计算交点
        P1 = self._line_intersection(l1, l2)
        P2 = self._line_intersection(l1, l3)

        if P1 is None or P2 is None:
            # 平行或数值稳定性问题，放弃扩展
            return mask

        # 判断“新衣是否更长”：检查 p6, p7 相对 l1 的位置
        # 图像坐标中，y 向下为正。如果我们希望“更长=在原下摆线之下”，
        # 可以这样判断：取人坐标系里“下方”为 y 更大。
        #
        # 对直线 A x + B y + C = 0，
        # 对两个点 h6, p6 来看，如果 sign(值) 不同，则在两侧；
        # 这里我们更简单直接比较 y 坐标：如果 p6.y > h6.y 且 p7.y > h7.y，则认为更长。
        if not (p6[1] > h6[1] and p7[1] > h7[1]):
            # 新衣在原衣服下摆线之上或差不多，不做扩展
            return mask

        # 构造四边形：P1, p6, p7, P2
        poly = np.array([
            [P1[0], P1[1]],
            [p6[0], p6[1]],
            [p7[0], p7[1]],
            [P2[0], P2[1]]
        ], dtype=np.float32)

        # 转为 int，限制在图像范围内
        poly_int = np.round(poly).astype(np.int32)
        poly_int[:, 0] = np.clip(poly_int[:, 0], 0, width - 1)
        poly_int[:, 1] = np.clip(poly_int[:, 1], 0, height - 1)

        cv.fillPoly(mask, [poly_int], 1)

        return mask
    def __init__(self, gmm_model, tom_model, backend, target, json_path=None):
        super(CpVton, self).__init__()
        # gmm_model 实际已不再使用，但保持加载不影响效果
        self.gmm_net = cv.dnn.readNet(gmm_model)
        self.tom_net = cv.dnn.readNet(tom_model)
        self.gmm_net.setPreferableBackend(backend)
        self.gmm_net.setPreferableTarget(target)
        self.tom_net.setPreferableBackend(backend)
        self.tom_net.setPreferableTarget(target)
        self.downsample = BilinearFilter()

        self.current_pair = None
        self.json_path = json_path or 'points_data.json'

    def prepare_agnostic(self, segm_image, input_image, pose_map, height=256, width=192):
        palette = {
            'Background': (0, 0, 0),
            'Hat': (128, 0, 0),
            'Hair': (255, 0, 0),
            'Glove': (0, 85, 0),
            'Sunglasses': (170, 0, 51),
            'UpperClothes': (255, 85, 0),
            'Dress': (0, 0, 85),
            'Coat': (0, 119, 221),
            'Socks': (85, 85, 0),
            'Pants': (0, 85, 85),
            'Jumpsuits': (85, 51, 0),
            'Scarf': (52, 86, 128),
            'Skirt': (0, 128, 0),
            'Face': (0, 0, 255),
            'Left-arm': (51, 170, 221),
            'Right-arm': (0, 255, 255),
            'Left-leg': (85, 255, 170),
            'Right-leg': (170, 255, 85),
            'Left-shoe': (255, 255, 0),
            'Right-shoe': (255, 170, 0)
        }
        color2label = {val: key for key, val in palette.items()}
        head_labels = ['Hat', 'Hair', 'Sunglasses', 'Face']

        segm_image = cv.cvtColor(segm_image, cv.COLOR_BGR2RGB)
        phead = np.zeros((1, height, width), dtype=np.float32)
        pose_shape = np.zeros((height, width), dtype=np.uint8)
        for r in range(height):
            for c in range(width):
                pixel = tuple(segm_image[r, c])
                if pixel in color2label:
                    if color2label[pixel] in head_labels:
                        phead[0, r, c] = 1
                    if color2label[pixel] != 'Background':
                        pose_shape[r, c] = 255

        input_blob = cv.dnn.blobFromImage(
            input_image, 1.0 / 127.5, (width, height),
            mean=(127.5, 127.5, 127.5), swapRB=True
        )
        input_blob = input_blob.squeeze(0)
        img_head = input_blob * phead - (1 - phead)

        down = self.downsample.imaging_resample(pose_shape, width // 16, height // 16)
        res_shape = cv.resize(down, (width, height), cv.INTER_LINEAR)

        res_shape = cv.dnn.blobFromImage(
            res_shape, 1.0 / 127.5, mean=(127.5, 127.5, 127.5), swapRB=True
        )
        res_shape = res_shape.squeeze(0)

        agnostic = np.concatenate((res_shape, img_head, pose_map), axis=0)
        agnostic = np.expand_dims(agnostic, axis=0)
        return agnostic.astype(np.float32)

    def load_pair_points(self, height, width):
        """
        从 JSON 加载特征点并缩放（用于可视化打点，不影响 warp_cloth）
        """
        try:
            with open(self.json_path, 'r') as f:
                pairs_data = json.load(f)

            current_pair = self.current_pair
            print(f"当前处理的Pair: {current_pair}")

            for pair_data in pairs_data:
                print(f"遍历Pair: {pair_data.get('pair')}")
                if pair_data.get('pair') == current_pair:
                    cloth_points = pair_data.get('cloth_points', [])
                    pair_points = pair_data.get('pair_points', [])

                    cloth_points = self.scale_points(cloth_points, height, width)
                    pair_points = self.scale_points(pair_points, height, width)

                    print(f"cloth_points数量: {len(cloth_points)}")
                    print(f"pair_points数量: {len(pair_points)}")

                    return {
                        'cloth_points': cloth_points,
                        'human_points': pair_points
                    }

            raise ValueError(f"未找到匹配的配对: {current_pair}")

        except FileNotFoundError:
            print(f"JSON文件未找到: {self.json_path}")
            return None
        except json.JSONDecodeError:
            print(f"JSON解析错误: {self.json_path}")
            return None

    def scale_points(self, points, height, width):
        original_width, original_height = 192, 256
        scaled_points = []
        for x, y in points:
            scaled_x = int(x * width / original_width)
            scaled_y = int(y * height / original_height)
            scaled_points.append([scaled_x, scaled_y])
        return scaled_points

    def get_warped_cloth(self, cloth_img, agnostic, height=256, width=192):
        # resized = cv.resize(cloth_img, (width, height))
        cloth_blob = cv.dnn.blobFromImage(
            cloth_img,
            1.0 / 127.5,
            (width, height),
            mean=(127.5, 127.5, 127.5),
            swapRB=True
        )
        return cloth_blob

    def build_cloth_region_mask(self, pair_points_scaled, height=256, width=192):
        """
        使用 pair_points_scaled 构造一个 soft 的衣服区域 mask:
        - 多边形内部为 1，外部为 0
        - 通过高斯模糊让边缘成为 [0,1] 平滑过渡
        pair_points_scaled: 已按 (width, height) 缩放的 12 个点，顺序为：
            左领口尖端, 左肩上, 左袖口上, 左袖口下,
            左腋下, 左下摆, 右下摆, 右腋下,
            右袖口下, 右袖口上, 右肩上, 右领口
        """
        # 默认：如果点异常，返回全 0，不做增强
        mask = np.zeros((height, width), dtype=np.float32)

        if pair_points_scaled is None or len(pair_points_scaled) < 3:
            return mask

        pts = np.array(pair_points_scaled, dtype=np.int32)

        # 填充多边形为1
        cv.fillPoly(mask, [pts], 1.0)

        # 高斯模糊软化边缘（核大小可以调整，越大越“宽”的边缘区域）
        mask = cv.GaussianBlur(mask, (31, 31), 0)
        mask = np.clip(mask, 0.0, 1.0)

        return mask  # shape: (H, W), 值在[0,1]

    def get_tryon(self, agnostic, warp_cloth,
                  pair_points_scaled=None, height=256, width=192):
        """
        利用 TOM 做最终 try-on。

        新增:
        - 如果提供了 pair_points_scaled，则在 pair_points 定义的衣服多边形
          区域内（尤其是靠近边缘的位置），适度放大 m_composite，
          让该区域更偏向 warp_cloth 的真实衣服纹理。
        - 多边形外区域的 m_composite 尽量保持原样，不做压低。
        """
        # 1. 前向推理
        inp = np.concatenate([agnostic, warp_cloth], axis=1)
        self.tom_net.setInput(inp)
        out = self.tom_net.forward()

        # 2. 拆分输出，并做激活
        p_rendered, m_composite = np.split(out, [3], axis=1)
        p_rendered = np.tanh(p_rendered)
        m_composite = 1 / (1 + np.exp(-m_composite))  # (1,1,H,W), in (0,1)

        # 3. 使用 pair_points_scaled 对 m_composite 做“局部增强”
        if pair_points_scaled is not None and len(pair_points_scaled) >= 3:
            # 3.1 构造 2D mask（H,W），多边形区域为1，外部为0，边缘有平滑过渡
            cloth_region_mask = self.build_cloth_region_mask(
                pair_points_scaled, height=height, width=width
            )  # (H, W) in [0,1]

            # 3.2 扩展为 (1,1,H,W) 以便与 m_composite 同形
            cloth_region_mask_4d = cloth_region_mask[np.newaxis, np.newaxis, :, :]

            # 3.3 定义一个“增强因子”：在多边形内部 >1，外部 ≈1
            #
            # 例如:
            #   boost_strength = 0.3 =>
            #     mask=1 的地方: boost = 1.3 (m 增大 30%)
            #     mask=0 的地方: boost = 1.0 (不变)
            #     mask 介于(0,1)时效果平滑过渡（尤其边缘更加强）
            boost_strength = 0.3  # 可调，0.2~0.5 之间试
            boost_factor = 1.0 + boost_strength * cloth_region_mask_4d

            # 3.4 应用增强，并裁剪到[0,1]，避免超过范围
            m_composite = m_composite * boost_factor
            m_composite = np.clip(m_composite, 0.0, 1.0)

        # 4. 按照 CP-VTON 的公式融合
        p_tryon = warp_cloth * m_composite + p_rendered * (1 - m_composite)

        # 5. 转回可显示的 RGB
        rgb_p_tryon = cv.cvtColor(
            p_tryon.squeeze(0).transpose(1, 2, 0), cv.COLOR_BGR2RGB
        )
        rgb_p_tryon = (rgb_p_tryon + 1) / 2
        return rgb_p_tryon


class CorrelationLayer(object):
    def __init__(self, params, blobs):
        super(CorrelationLayer, self).__init__()

    def getMemoryShapes(self, inputs):
        featureAShape = inputs[0]
        b, _, h, w = featureAShape
        return [[b, h * w, h, w]]

    def forward(self, inputs):
        b, c, h, w = inputs[0].shape
        feature_A = inputs[0].transpose(0, 3, 2, 1)
        feature_B = inputs[1].transpose(0, 2, 3, 1)
        feature_A = feature_A.reshape(1, -1, 512)
        feature_B = feature_B.reshape(1, -1, 512)

        feature_A = feature_A.transpose(0, 2, 1)
        feature_mul = feature_B @ feature_A

        feature_mul = feature_mul.reshape((b, h, w, h * w))
        correlation_tensor = feature_mul.transpose((0, 3, 1, 2))
        correlation_tensor = np.ascontiguousarray(correlation_tensor)
        return [correlation_tensor]


def draw_points(image, points, color=(0, 255, 0), radius=5):
    image_with_points = image.copy()
    for point in points:
        x, y = map(int, point)
        cv.circle(image_with_points, (x, y), radius, color, -1)
    return image_with_points


if __name__ == "__main__":
    try:
        for attr in (
            'input_image', 'input_cloth', 'gmm_model', 'tom_model',
            'segmentation_model', 'openpose_proto', 'openpose_model'
        ):
            setattr(args, attr, resolve_path(getattr(args, attr)))

        json_paths = [
            os.path.join(PROJECT_ROOT, 'data/final_combined_pairs.json'),
            os.path.join(PROJECT_ROOT, 'points_data.json'),
            'points_data.json'
        ]

        json_path = next((path for path in json_paths if os.path.exists(path)), None)
        if not json_path:
            raise FileNotFoundError("未找到JSON配对文件")

        with open(json_path, 'r') as f:
            pairs_data = json.load(f)

        # 取第一项
        first_pair = pairs_data[0]
        pair_name = first_pair['pair']

        # 构建图片路径
        cloth_filename = pair_name.split('_')[0] + '_c.jpg'
        human_filename = pair_name.split('_')[2] + '_h.jpg'

        # 多路径图片查找
        cloth_paths = [
            os.path.join(PROJECT_ROOT, 'data/raw_images/clothes', cloth_filename),
            cloth_filename
        ]
        human_paths = [
            os.path.join(PROJECT_ROOT, 'data/raw_images/humans', human_filename),
            human_filename
        ]

        cloth_path = next((path for path in cloth_paths if os.path.exists(path)), None)
        human_path = next((path for path in human_paths if os.path.exists(path)), None)
        if not cloth_path or not human_path:
            raise FileNotFoundError("未找到匹配的图片文件")

        # 更新参数
        args.input_image = human_path
        args.input_cloth = cloth_path

        # 模型文件检查
        model_files = [
            ('gmm_model', args.gmm_model),
            ('tom_model', args.tom_model),
            ('segmentation_model', args.segmentation_model),
            ('openpose_proto', args.openpose_proto),
            ('openpose_model', args.openpose_model)
        ]
        for name, path in model_files:
            if not os.path.isfile(path):
                raise OSError(f"{name} 模型文件不存在: {path}")

        # 读取图像
        person_img = cv.imread(args.input_image)
        if person_img is None:
            raise ValueError(f"无法读取人物图像: {args.input_image}")

        cloth_img = cv.imread(args.input_cloth)
        if cloth_img is None:
            raise ValueError(f"无法读取服装图像: {args.input_cloth}")

        # 图像比例调整为 256x192 比例
        ratio = 256 / 192
        inp_h, inp_w, _ = person_img.shape
        current_ratio = inp_h / inp_w

        if current_ratio > ratio:
            center_h = inp_h // 2
            out_h = int(inp_w * ratio)
            start = int(center_h - out_h // 2)
            end = int(center_h + out_h // 2)
            person_img = person_img[start:end, ...]
        else:
            center_w = inp_w // 2
            out_w = int(inp_h / ratio)
            start = int(center_w - out_w // 2)
            end = int(center_w + out_w // 2)
            person_img = person_img[:, start:end, :]

        # 姿态估计
        pose = get_pose_map(
            person_img,
            args.openpose_proto,
            args.openpose_model,
            args.backend,
            args.target
        )

        # 人体分割
        segm_image = parse_human(person_img, args.segmentation_model)
        segm_image = cv.resize(segm_image, (192, 256), cv.INTER_LINEAR)

        # 注册关联层（GMM 结构需要，但现在GMM输出不被使用，保持兼容）
        cv.dnn_registerLayer('Correlation', CorrelationLayer)

        # 创建模型实例
        model = CpVton(
            args.gmm_model,
            args.tom_model,
            args.backend,
            args.target,
            json_path=json_path
        )
        model.current_pair = pair_name

        # 加载特征点用于可视化
        cloth_points = first_pair.get('cloth_points', [])
        pair_points = first_pair.get('pair_points', [])
        human_points = first_pair.get('human_points', [])

        cloth_points_scaled = model.scale_points(cloth_points, 256, 192)
        pair_points_scaled = model.scale_points(pair_points, 256, 192)
        human_points_scaled = model.scale_points(human_points, 256, 192)

        aligned_cloth_img, warped_cloth_points_scaled = align_and_warp_cloth(
            cloth_img,
            cloth_points_scaled,
            pair_points_scaled,
            bg_color=(230, 230, 230)
        )
        cv.imshow("", aligned_cloth_img)
        cloth_img = aligned_cloth_img
        cloth_points_scaled = warped_cloth_points_scaled.tolist()

        # 推理流程
        agnostic = model.prepare_agnostic(segm_image, person_img, pose)
        hem_mask_dbg = model.build_extended_hem_mask(
            human_points_scaled,
            pair_points_scaled,
            height=256,
            width=192
        )
        cv.imshow("hem_mask", hem_mask_dbg * 255)
        cv.waitKey(0)
        agnostic_for_tom = model.extend_agnostic_with_hem(
            agnostic,
            human_points_scaled=human_points_scaled,
            pair_points_scaled=pair_points_scaled,
            height=256,
            width=192
        )
        warped_cloth = model.get_warped_cloth(cloth_img, agnostic_for_tom)

        output = model.get_tryon(
            agnostic_for_tom,
            warped_cloth,
            pair_points_scaled=pair_points_scaled,  # 新增
            height=256,
            width=192
        )

        # 取消注册关联层
        cv.dnn_unregisterLayer('Correlation')

        # 标注点
        person_img_with_points = draw_points(
            person_img, pair_points_scaled, color=(0, 0, 255), radius=8
        )
        cloth_img_with_points = draw_points(
            cloth_img, cloth_points_scaled, color=(255, 0, 0), radius=8
        )
        output_with_points = draw_points(
            output, pair_points_scaled, color=(0, 0, 255), radius=5
        )
        output_with_points = draw_points(
            output_with_points, cloth_points_scaled, color=(255, 0, 0), radius=5
        )

        # 显示结果
        cv.imshow('Person Image with Points', person_img_with_points)
        cv.imshow('Cloth Image with Points', cloth_img_with_points)
        cv.imshow('Virtual Try-On with Points', output_with_points)
        cv.waitKey(0)
        cv.destroyAllWindows()

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()