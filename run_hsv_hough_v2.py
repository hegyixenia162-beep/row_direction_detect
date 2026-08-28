"""HSV + Hough 种植行检测 V2：针对两张大图修正旋转、分区多方向并增强弱苗。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import run_hsv_hough as v1


def parse_args() -> argparse.Namespace:
    """解析V2参数；默认只处理人工标注过的两张大图，避免混入五张暂不优化的小图。"""

    parser = argparse.ArgumentParser(description="HSV + Hough 棉花种植行检测 V2")
    parser.add_argument("--input", default="test_photo", help="输入图像目录")
    parser.add_argument("--output", default="outputs_v2", help="V2结果输出目录")
    parser.add_argument("--names", nargs="+", default=["photo1.jpg", "photo2.jpg"], help="仅处理的图像文件名")
    parser.add_argument("--max-side", type=int, default=1800, help="工作图最长边")
    parser.add_argument("--h-min", type=int, default=25, help="强植被HSV色相下限")
    parser.add_argument("--h-max", type=int, default=100, help="强植被HSV色相上限")
    parser.add_argument("--s-min", type=int, default=35, help="强植被饱和度下限")
    parser.add_argument("--v-min", type=int, default=25, help="强植被亮度下限")
    parser.add_argument("--min-component", type=int, default=18, help="强植被最小连通域面积")
    parser.add_argument("--angle-tolerance", type=float, default=12.0, help="Hough主方向聚类容差")
    return parser.parse_args()


def build_enhanced_mask(image: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """生成强植被掩膜及弱苗增强掩膜；弱响应必须同时满足绿色优势，避免单纯放宽HSV。"""

    strong = v1.build_hsv_mask(image, args)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(image)
    green_i = green.astype(np.int16)
    red_i = red.astype(np.int16)
    blue_i = blue.astype(np.int16)
    excess_green = 2 * green_i - red_i - blue_i

    # 弱苗候选允许更低饱和度和亮度，但必须保持绿色通道优势，为什么：弱小叶片像素常被土壤混色。
    weak_condition = (
        (hsv[:, :, 0] >= max(15, args.h_min - 10))
        & (hsv[:, :, 0] <= min(115, args.h_max + 10))
        & (hsv[:, :, 1] >= max(20, args.s_min - 10))
        & (hsv[:, :, 2] >= max(15, args.v_min - 10))
        & (excess_green >= 30)
        & ((green_i - red_i) >= 8)
    )
    weak = np.where(weak_condition, 255, 0).astype(np.uint8)
    weak = cv2.morphologyEx(weak, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    weak_clean = np.zeros_like(weak)
    # 弱苗面积门限小于强掩膜门限，使原本被面积过滤掉的小植株仍能作为行支持证据。
    for label in range(1, count):
        # 仅保留至少6个工作尺度像素的弱绿色连通域，删除单像素色噪声。
        if stats[label, cv2.CC_STAT_AREA] >= 6:
            weak_clean[labels == label] = 255
    # 弱苗只在强植被附近补充；9×9邻域可连接叶缘和小苗碎片，又不会吞入整片偏绿色背景。
    strong_corridor = cv2.dilate(strong, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    weak_near_rows = cv2.bitwise_and(weak_clean, strong_corridor)
    enhanced = cv2.bitwise_or(strong, weak_near_rows)
    return strong, enhanced


def corrected_rotation_matrix(shape: tuple[int, int], angle_deg: float) -> tuple[np.ndarray, tuple[int, int]]:
    """把行轴旋转到竖直方向；修正V1中使用相反符号导致92°被变成约88°的问题。"""

    height, width = shape
    center = (width / 2.0, height / 2.0)
    # OpenCV正角为逆时针；轴角angle要变到90°，图像内容应旋转angle-90，而不是90-angle。
    rotation_deg = angle_deg - 90.0
    matrix = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(height * sine + width * cosine)
    new_height = int(height * cosine + width * sine)
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    return matrix, (new_width, new_height)


def projection_periodicity_score(mask: np.ndarray, angle_deg: float) -> float:
    """计算给定方向法向上的植被密度起伏，用于区分真实行方向与叶缘伪Hough方向。"""

    height, width = mask.shape
    yy, xx = np.indices((height, width))
    radians = math.radians(angle_deg)
    normal_x, normal_y = -math.sin(radians), math.cos(radians)
    coordinate = xx * normal_x + yy * normal_y
    minimum = math.floor(float(coordinate.min()))
    bin_index = np.floor(coordinate - minimum).astype(np.int32).ravel()
    domain_count = np.bincount(bin_index)
    vegetation_count = np.bincount(bin_index, weights=(mask > 0).ravel(), minlength=len(domain_count))
    density = vegetation_count / np.maximum(domain_count, 1)
    density = gaussian_filter1d(density, sigma=2.0)
    valid = domain_count > 0.05 * max(height, width)
    density = density[valid]
    # 有效投影长度不足时返回0，防止窄条带产生虚假的高方差。
    if len(density) < 20 or float(np.mean(density)) <= 1e-8:
        return 0.0
    return float(np.std(density) / np.mean(density))


def detect_direction_regions(strong_mask: np.ndarray, global_angle: float) -> list[tuple[int, int, float]]:
    """检测上下方向区域；连续出现正交周期时建立一次方向切换，适配photo1而不误分photo2。"""

    height = strong_mask.shape[0]
    strip_count = 12
    orthogonal_angle = (global_angle + 90.0) % 180.0
    orthogonal_votes: list[bool] = []
    # 逐水平条带比较全局方向和正交方向的周期性；不采用局部Hough，因为叶缘短线会误导方向。
    for strip_index in range(strip_count):
        y0 = int(strip_index * height / strip_count)
        y1 = int((strip_index + 1) * height / strip_count)
        strip = strong_mask[y0:y1]
        global_score = projection_periodicity_score(strip, global_angle)
        orthogonal_score = projection_periodicity_score(strip, orthogonal_angle)
        orthogonal_votes.append(orthogonal_score > 1.15 * max(global_score, 1e-8))

    switch_index: int | None = None
    # 至少两个连续条带支持正交方向才切换，为什么：单个异常条带可能来自地头或局部缺苗。
    for index in range(strip_count - 1):
        # 找到首次稳定方向切换后停止，本版针对两张大图的一次上下区域转换，不扩展到任意多地块。
        if orthogonal_votes[index] and orthogonal_votes[index + 1]:
            switch_index = index
            break
    # 没有稳定正交区时整图保持单方向，photo2应进入此分支。
    if switch_index is None:
        return [(0, height, global_angle)]

    boundary = int((switch_index + 0.5) * height / strip_count)
    return [(0, boundary, global_angle), (boundary, height, orthogonal_angle)]


def detect_rows_in_region(
    mask: np.ndarray,
    angle_deg: float,
    y_offset: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray, int, float]], int]:
    """在单一方向区域内检测投影峰并拟合有限线段，最后把局部坐标映射回工作图。"""

    matrix, output_size = corrected_rotation_matrix(mask.shape, angle_deg)
    rotated = cv2.warpAffine(mask, matrix, output_size, flags=cv2.INTER_NEAREST, borderValue=0)
    raw_profile = np.sum(rotated > 0, axis=0).astype(np.float64)
    smooth_profile = gaussian_filter1d(raw_profile, sigma=max(1.5, 0.002 * len(raw_profile)))
    spacing = v1.estimate_spacing(smooth_profile)
    prominence = max(float(np.max(smooth_profile)) * 0.06, float(np.std(smooth_profile)) * 0.30)
    peaks, _ = find_peaks(
        smooth_profile,
        distance=max(8, int(0.55 * spacing)),
        prominence=prominence,
    )

    inverse = cv2.invertAffineTransform(matrix)
    segments: list[tuple[np.ndarray, np.ndarray, int, float]] = []
    half_width = max(4, int(0.28 * spacing))
    # 每个投影峰只产生一条中心线；本轮只针对大图，不处理小图的畦内双行问题。
    for peak in peaks:
        points = v1.sample_row_points(rotated, int(peak), half_width)
        # 大图行应具有长程支持；点数和跨度不足的弱峰不输出，杂草问题按用户要求暂不专项优化。
        if len(points) < 12 or np.ptp(points[:, 1]) < 0.18 * rotated.shape[0]:
            continue
        vx, vy, x0, y0 = (float(value) for value in cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1))
        # 正确旋转后代表点应接近竖直；偏斜过大时使用中位横坐标，防止局部弱苗拉斜整行。
        if abs(vx) > 0.25 or abs(vy) < 1e-6:
            vx, vy, x0 = 0.0, 1.0, float(np.median(points[:, 0]))
        y_start, y_end = np.quantile(points[:, 1], [0.02, 0.98])
        start_rotated = np.array([x0 + vx / vy * (y_start - y0), y_start, 1.0], dtype=np.float64)
        end_rotated = np.array([x0 + vx / vy * (y_end - y0), y_end, 1.0], dtype=np.float64)
        start = inverse @ start_rotated
        end = inverse @ end_rotated
        # 局部区域坐标只需补回纵向偏移；横坐标已经处于完整工作图坐标系。
        start[1] += y_offset
        end[1] += y_offset
        coverage = float(np.ptp(points[:, 1]) / max(1, rotated.shape[0]))
        segments.append((start, end, len(points), coverage))
    return segments, spacing


def process_image(path: Path, output_dir: Path, args: argparse.Namespace) -> dict:
    """处理一张大图，输出增强掩膜、分方向有限线段呈现图及结构化JSON。"""

    original = cv2.imread(str(path), cv2.IMREAD_COLOR)
    # 图像解码失败时明确终止该图处理，避免输出空成功记录。
    if original is None:
        raise RuntimeError(f"无法读取图像：{path}")
    work, scale = v1.resize_for_work(original, args.max_side)
    strong_mask, enhanced_mask = build_enhanced_mask(work, args)
    mask_ratio = float(np.mean(enhanced_mask > 0))
    # 掩膜占比异常意味着弱苗增强条件失控或未检测到植被，此时拒绝继续出线。
    if not 0.002 <= mask_ratio <= 0.65:
        raise RuntimeError(f"V2增强掩膜占比异常：{mask_ratio:.4f}")

    global_angle, hough_support = v1.estimate_hough_angle(strong_mask, args.angle_tolerance)
    regions = detect_direction_regions(strong_mask, global_angle)
    all_segments: list[tuple[np.ndarray, np.ndarray, int, float]] = []
    region_records: list[dict] = []
    # 分方向区域独立检测，photo1的纵向行不会再被下方横向植被错误延伸。
    for region_id, (y0, y1, angle) in enumerate(regions, start=1):
        segments, spacing = detect_rows_in_region(enhanced_mask[y0:y1], angle, y0)
        all_segments.extend(segments)
        region_records.append(
            {
                "region_id": region_id,
                "y_range_work": [y0, y1],
                "angle_deg": angle,
                "estimated_row_spacing_px": spacing / scale,
                "row_count_before_final_filter": len(segments),
            }
        )

    overlay, rows = v1.draw_and_serialize(original, all_segments, scale)
    # 两张大图都应存在可靠行；完全没有结果视为失败。
    if not rows:
        raise RuntimeError("V2未获得满足支持度与长度要求的种植行")
    mask_full = cv2.resize(enhanced_mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_path = output_dir / f"{path.stem}_mask_v2.png"
    overlay_path = output_dir / f"{path.stem}_rows_v2.png"
    json_path = output_dir / f"{path.stem}_rows_v2.json"
    cv2.imwrite(str(mask_path), mask_full)
    cv2.imwrite(str(overlay_path), overlay)
    result = {
        "version": "v2",
        "source": str(path),
        "image_size": {"width": original.shape[1], "height": original.shape[0]},
        "work_scale": scale,
        "enhanced_mask_ratio": mask_ratio,
        "global_hough_angle_deg": global_angle,
        "hough_support_lines": hough_support,
        "direction_region_count": len(regions),
        "direction_regions": region_records,
        "row_count": len(rows),
        "rows": [asdict(row) for row in rows],
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    """只批处理参数指定的两张大图，并生成V2独立汇总，不覆盖V1结果。"""

    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    # 按显式文件名处理，确保后续对比范围始终只有photo1和photo2。
    for name in args.names:
        path = input_dir / name
        try:
            result = process_image(path, output_dir, args)
            summary.append({"source": str(path), "status": "success", "row_count": result["row_count"]})
            print(f"[V2成功] {name}: {result['row_count']} 行，{result['direction_region_count']} 个方向区域")
        except Exception as exc:
            summary.append({"source": str(path), "status": "failed", "error": str(exc)})
            print(f"[V2失败] {name}: {exc}")
    (output_dir / "run_summary_v2.json").write_text(
        json.dumps({"parameters": vars(args), "images": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# 直接执行时运行两张大图；作为模块导入时不产生输出文件。
if __name__ == "__main__":
    main()
