import cv2
import numpy as np
import cv2 as cv


IMG_WIDTH = 192
IMG_HEIGHT = 256

def clean_cloth_borders(cloth_img, margin=20, bg_color=(230, 230, 230)):
    """
    将 cloth_img 四周 margin 像素清成 bg_color。
    用于在整体平移前，把衣服图像边沿（通常是黑边）清理掉。
    """
    if cloth_img is None:
        return cloth_img

    img = cloth_img.copy()
    h, w = img.shape[:2]

    m = int(margin)
    if m <= 0:
        return img
    m = max(1, min(m, h // 2, w // 2))

    # 上
    img[:m, :, :] = bg_color
    # 下
    img[h - m:h, :, :] = bg_color
    # 左
    img[:, :m, :] = bg_color
    # 右
    img[:, w - m:w, :] = bg_color

    return img

def _ensure_shape(cloth_img):
    """保证衣服图像是 256x192，大于则 resize，小于则居中贴到背景."""
    h, w = cloth_img.shape[:2]
    if (w, h) == (IMG_WIDTH, IMG_HEIGHT):
        return cloth_img

    # 简单策略：先等比缩放到不超过 192x256，然后居中贴到背景
    scale = min(IMG_WIDTH / w, IMG_HEIGHT / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv.resize(cloth_img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
    canvas = np.full((IMG_HEIGHT, IMG_WIDTH, 3), 230, dtype=np.uint8)
    x0 = (IMG_WIDTH - new_w) // 2
    y0 = (IMG_HEIGHT - new_h) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas


def translate_cloth_to_pair(cloth_img, cloth_pts, pair_pts, bg_color=(230, 230, 230)):
    """
    第一部分：整体平移（不拉伸）
    cloth_pts, pair_pts: (12,2) float32，基于 192x256 的坐标
    返回：平移后的图像 & 点
    """
    # 确保图像形状
    cloth_img = _ensure_shape(cloth_img)
    h, w = cloth_img.shape[:2]

    cloth_pts = np.asarray(cloth_pts, dtype=np.float32)
    pair_pts = np.asarray(pair_pts, dtype=np.float32)

    # A 区的点索引: 1,2,5,8,11,12 -> 0,1,4,7,10,11
    torso_idx = np.array([0, 1, 4, 7, 10, 11], dtype=np.int32)

    src_torso = cloth_pts[torso_idx]
    dst_torso = pair_pts[torso_idx]

    delta = np.mean(dst_torso - src_torso, axis=0)
    dx, dy = np.round(delta).astype(int)

    # 构造平移后的画布
    canvas_height = h
    if dy > 0:  # 如果图像在垂直方向上向下移动，扩展画布高度
        canvas_height += dy
    canvas = np.full((canvas_height, w, 3), bg_color, dtype=np.uint8)

    # 计算有效 ROI
    x1_src = max(0, -dx)
    y1_src = max(0, -dy)
    x2_src = min(w, w - dx) if dx > 0 else min(w, w + dx)

    # Adjust y2_src based on dy (which is the vertical shift)
    y2_src = min(h, h + dy) if dy > 0 else min(h, h - dy)

    x1_dst = x1_src + dx
    y1_dst = y1_src + dy
    x2_dst = x2_src + dx
    y2_dst = y2_src + dy

    # 确保源区域合理
    if x2_src > x1_src and y2_src > y1_src:
        # 注意到 canvas 的高度可能已经增大，因此直接从原图中获取正确的区域
        canvas[y1_dst:y2_dst, x1_dst:x2_dst] = cloth_img[y1_src:y2_src, x1_src:x2_src]

    # 更新点坐标
    new_pts = cloth_pts.copy()
    new_pts[:, 0] += dx
    new_pts[:, 1] += dy

    new_pts[:, 0] = np.clip(new_pts[:, 0], 0, w - 1)
    new_pts[:, 1] = np.clip(new_pts[:, 1], 0, canvas_height - 1)  # 裁剪到新画布的高度

    return canvas, new_pts


def _warp_quad_region(base_img, src_quad, dst_quad,
                      bg_color=(230, 230, 230),
                      erase_src=True):
    """
    使用透视变换将一个四边形区域从 src_quad 变到 dst_quad。
    base_img: 当前画布
    src_quad, dst_quad: shape (4,2) float32
    返回：变形后的图像, 3x3 透视矩阵 H
    """
    h, w = base_img.shape[:2]
    src_quad = np.asarray(src_quad, dtype=np.float32)
    dst_quad = np.asarray(dst_quad, dtype=np.float32)

    H = cv.getPerspectiveTransform(src_quad, dst_quad)

    # 把整张图像按 H warp 到新图上
    warped = cv.warpPerspective(
        base_img, H, (w, h),
        flags=cv.INTER_LINEAR,
        borderValue=bg_color
    )

    # src 多边形 mask：用于在 base_img 中抹掉原区域
    src_mask = np.zeros((h, w), dtype=np.uint8)
    cv.fillConvexPoly(src_mask, src_quad.astype(np.int32), 255)

    # dst 多边形 mask：用于在 warped 中抠出新区域覆盖到 base_img
    dst_mask = np.zeros((h, w), dtype=np.uint8)
    cv.fillConvexPoly(dst_mask, dst_quad.astype(np.int32), 255)

    result = base_img.copy()
    if erase_src:
        result[src_mask > 0] = bg_color

    # 覆盖新区域
    mask3 = dst_mask.astype(bool)
    result[mask3] = warped[mask3]

    return result, H


def _apply_homography_to_points(points, idxs, H):
    """
    对 points[idxs] 应用透视变换 H，更新这些点。
    points: (N,2) float32
    idxs: 一维索引数组
    """
    pts = points[idxs].astype(np.float32)
    # 转为齐次坐标 (x, y, 1)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)  # (n,3)
    pts_h = pts_h @ H.T  # (n,3)
    # 除以最后一维
    pts_new = pts_h[:, :2] / np.maximum(pts_h[:, 2:3], 1e-8)
    points[idxs] = pts_new
    return points


def warp_region_C(cloth_img, cloth_pts, pair_pts, bg_color=(230, 230, 230), hem_offset=5):
    h, w = cloth_img.shape[:2]
    pts = cloth_pts.copy()

    # 源骨架点
    p5_src = pts[4].astype(np.float32)  # 左腋下
    p6_src = pts[5].astype(np.float32)  # 左衣服下边沿
    p7_src = pts[6].astype(np.float32)  # 右衣服下边沿
    p8_src = pts[7].astype(np.float32)  # 右腋下

    # 计算新坐标 P1 和 P2（根据 hem_offset 下移）
    P1 = np.array([p6_src[0], p6_src[1] + hem_offset], dtype=np.float32)
    P2 = np.array([p7_src[0], p7_src[1] + hem_offset], dtype=np.float32)

    # 定义变形区域的四个顶点
    src_pts = np.array([p5_src, P1, P2, p8_src], dtype=np.float32)

    # 目标区域的四个顶点（p5_src 和 p8_src 固定，更新 P1 和 P2 的位置）
    target_p6 = pair_pts[5].astype(np.float32)  # 目标左衣服下边沿
    target_p7 = pair_pts[6].astype(np.float32)  # 目标右衣服下边沿

    # 更新目标 P1 和 P2 的 Y 坐标
    new_Y_P1 = target_p6[1]  # 目标 Y
    new_Y_P2 = target_p7[1]  # 目标 Y
    target_pts = np.array([p5_src, [p6_src[0], new_Y_P1], [p7_src[0], new_Y_P2], p8_src], dtype=np.float32)

    # 计算仿射变换矩阵
    matrix = cv2.getPerspectiveTransform(src_pts, target_pts)

    # 创建一个与原图像相同的空白图像
    cloth_warped = np.full(cloth_img.shape, bg_color, dtype=np.uint8)

    # 进行变形以得到变形区域
    transformed_region = cv2.warpPerspective(cloth_img, matrix, (w, h), flags=cv2.INTER_LINEAR)

    # 创建掩码以填充变形区域
    mask = np.zeros(cloth_img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(src_pts), 255)

    # 将变形区域的内容合并到新的背景图像中
    cloth_warped[mask == 255] = transformed_region[mask == 255]

    # 确保变形区域下部产生的空隙用背景色填充
    # 只需将药物掩码应用于变形完后的总图像
    cloth_combined = np.where(mask[:, :, None].astype(bool), cloth_warped, cloth_img)

    # 更新点 6 和 7 的位置
    new_pts = pts.copy()
    new_pts[5] = [p6_src[0], new_Y_P1]  # 更新变形后的点
    new_pts[6] = [p7_src[0], new_Y_P2]  # 更新变形后的点

    # 返回变形后的完整衣服图像和所有更新后的点
    return cloth_combined, new_pts

def warp_sleeve_region(cloth_img, cloth_pts, pair_pts,
                       idx_quad, left_or_right='left',
                       bg_color=(230, 230, 230),
                       anchor_weight=0.3):
    """
    通用袖子变形函数：
    idx_quad: 四个点索引 (肩上, 袖口上, 袖口下, 腋下) 的 0-based 索引
      左袖: [1,2,3,4]  对应点 2,3,4,5
      右袖: [7,8,9,10] 对应点 8,9,10,11

    规则：
    - 连接躯干的两点（肩上, 腋下）变形权重较小：
        dest = (1 - w) * current + w * pair
      w = anchor_weight (如 0.3)
    - 袖口两点（袖口上, 袖口下）直接对齐 pair（w = 1.0）
    - 旧袖子区域用 bg_color 抹掉，新袖子按照四边形透视 warp 生成；
      透视内部插值是布料纹理，不会“流出”到外面，旧区域直接白底。
    """
    h, w = cloth_img.shape[:2]
    pts = cloth_pts.copy()

    idx_quad = np.array(idx_quad, dtype=np.int32)
    src_quad = pts[idx_quad]

    # 源顺序: [肩上, 袖口上, 袖口下, 腋下]
    # pair 中对应的目标位置
    pair_quad = pair_pts[idx_quad]

    dst_quad = np.zeros_like(src_quad)
    # 0: 肩上, 3: 腋下 -> anchor
    for i in [0, 3]:
        dst_quad[i] = (1.0 - anchor_weight) * src_quad[i] + anchor_weight * pair_quad[i]
    # 1: 袖口上, 2: 袖口下 -> 完全对齐 pair
    dst_quad[1] = pair_quad[1]
    dst_quad[2] = pair_quad[2]

    warped_img, H = _warp_quad_region(
        cloth_img,
        src_quad=src_quad,
        dst_quad=dst_quad,
        bg_color=bg_color,
        erase_src=True
    )

    pts = _apply_homography_to_points(pts, idx_quad, H)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return warped_img, pts

def fix_and_crop(img_c, pts_c, bg_color=(230, 230, 230), temp=5,target_height=256):
    # 找到 pts_c 中最大的 y 坐标值
    max_y = int(np.max(pts_c[:, 1]))  # 转换为整数

    # 定义要处理的 y 范围，从 max_y 开始到图像底部
    y_start = min(max_y + temp, img_c.shape[0])

    # 将从最大 y 值到图像底部的所有像素设置为背景色
    img_c[y_start:, :] = bg_color  # 从 max_y 开始到最后一行

    if img_c.shape[0] > target_height:
        img_c = img_c[:target_height, :, :]  # 只保留上部分

    return img_c

def align_and_warp_cloth(cloth_img,
                         cloth_points_scaled,
                         pair_points_scaled,
                         bg_color=(230, 230, 230)):
    """
    主入口（当前版本）：
    - Step1: 整体平移
    - Step2: 只做下摆 C 区变形
    """
    cloth_img = _ensure_shape(cloth_img)

    # ① 在整体平移前，清理衣服图片边沿 20 像素为 bg_color
    cloth_img = clean_cloth_borders(cloth_img, margin=20, bg_color=bg_color)

    cloth_pts = np.asarray(cloth_points_scaled, dtype=np.float32).reshape(-1, 2)
    pair_pts = np.asarray(pair_points_scaled, dtype=np.float32).reshape(-1, 2)

    # Step 1: 整体平移（A 区为主）
    img_t, pts_t = translate_cloth_to_pair(
        cloth_img, cloth_pts, pair_pts, bg_color=bg_color
    )

    # Step 2-C: 下摆骨架 + 下摆带
    img_c, pts_c = warp_region_C(
        img_t, pts_t, pair_pts, bg_color=bg_color, hem_offset=5
    )
    # Step 3: 修复+裁剪
    img_fixed = fix_and_crop(img_c, pts_c, bg_color=bg_color)

    # 袖子暂不变形
    return img_fixed, pts_c