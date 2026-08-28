"""独立版 HSV + Hough 棉花种植行检测：合并 V1 基础能力与 V2 增强能力。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


PROJECT_ROOT = Path(__file__).resolve().parent
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class RowSegment:
    """保存一条种植行在原图坐标系中的有限线段和质量指标。"""

    row_id: int
    start: list[int]
    end: list[int]
    angle_deg: float
    length_px: float
    support_points: int
    longitudinal_coverage: float


def parse_args() -> argparse.Namespace:
    """解析命令行参数，使输入图片和阈值可以复现实验而不必修改源码。"""

    parser = argparse.ArgumentParser(description="独立版 HSV + Hough 棉花种植行检测")
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "test_photo"),
        help="单张输入图像或图像目录；默认是项目内 test_photo",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs_latest"),
        help="结果输出目录；默认是项目内 outputs_latest",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="目录模式下需要处理的文件名；省略时自动扫描目录内全部支持图像",
    )
    parser.add_argument("--max-side", type=int, default=1800, help="工作图最长边")
    parser.add_argument("--h-min", type=int, default=25, help="强植被 HSV 色相下限")
    parser.add_argument("--h-max", type=int, default=100, help="强植被 HSV 色相上限")
    parser.add_argument("--s-min", type=int, default=35, help="强植被饱和度下限")
    parser.add_argument("--v-min", type=int, default=25, help="强植被亮度下限")
    parser.add_argument("--min-component", type=int, default=18, help="强植被最小连通域面积")
    parser.add_argument("--angle-tolerance", type=float, default=12.0, help="Hough 主方向聚类容差")
    return parser.parse_args()


def configure_console_encoding() -> None:
    """尽量把终端输出设为 UTF-8，使 Windows 重定向或旧控制台也能正确显示中文。"""

    import sys

    # 标准流支持 reconfigure 时显式使用 UTF-8，为什么：Windows 控制台代码页可能造成中文日志乱码。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # 错误流也保持相同编码，为什么：异常信息和正常日志应采用一致的可读格式。
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def collect_images(input_path: Path, names: list[str] | None) -> list[Path]:
    """从单文件或目录收集输入图像，并以稳定顺序返回绝对路径。"""

    input_path = input_path.expanduser().resolve()
    # 单文件模式直接返回该图片，为什么：这是初次使用者最简单且最不容易写错的调用方式。
    if input_path.is_file():
        # 文件扩展名必须受支持，为什么：避免把 JSON 或其他文件误送给 OpenCV 解码。
        if input_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise RuntimeError(f"不支持的图像格式：{input_path.suffix or '无扩展名'}")
        # 单文件与 --names 同时出现含义冲突，为什么：names 只用于从目录选择文件。
        if names:
            raise RuntimeError("--input 指向单张图片时不能同时使用 --names")
        return [input_path]

    # 输入既不是文件也不是目录时明确报错，为什么：比后续空结果更容易定位路径问题。
    if not input_path.is_dir():
        raise RuntimeError(f"输入路径不存在或不是有效文件/目录：{input_path}")

    # 用户给出 names 时保持其顺序，为什么：显式顺序便于重复实验和逐图核对日志。
    if names:
        images = [input_path / name for name in names]
        missing = [path.name for path in images if not path.is_file()]
        # 任一指定文件不存在就停止，为什么：悄悄跳过会让批次范围与用户预期不一致。
        if missing:
            raise RuntimeError(f"指定图像不存在：{', '.join(missing)}")
        unsupported = [path.name for path in images if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES]
        # 指定文件格式不支持时停止，为什么：提前给出清晰错误比 OpenCV 解码失败更有帮助。
        if unsupported:
            raise RuntimeError(f"指定文件格式不受支持：{', '.join(unsupported)}")
        return images

    # 未给 names 时扫描目录并排序，为什么：让同一目录的多次运行具有完全一致的处理顺序。
    images = sorted(path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)
    # 空目录主动报错，为什么：仓库默认不附带航片，需提醒用户先放入自己的图像。
    if not images:
        raise RuntimeError(f"输入目录中没有支持的图像：{input_path}")
    return images


def resize_for_work(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """等比例缩小大图并返回缩放比，降低计算量且避免无意义地放大小图。"""

    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    # 只有原图超过工作尺寸时才缩小，为什么：放大小图会制造插值纹理却不会增加真实信息。
    if scale < 1.0:
        work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        work = image.copy()
    return work, scale


def build_hsv_mask(image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """使用 HSV 阈值、形态学和面积过滤生成可靠的强植被掩膜。"""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([args.h_min, args.s_min, args.v_min], dtype=np.uint8)
    upper = np.array([args.h_max, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    # 从标签 1 开始遍历，为什么：标签 0 是背景，不属于任何植被候选。
    for label in range(1, count):
        # 只保留面积足够的连通域，为什么：孤立小色块更可能是压缩噪声或土壤杂色。
        if stats[label, cv2.CC_STAT_AREA] >= args.min_component:
            cleaned[labels == label] = 255
    return cleaned


def axial_difference_deg(a: np.ndarray | float, b: float) -> np.ndarray:
    """计算无方向直线的最小夹角，使 0° 与 180° 被视为同一轴向。"""

    # 先平移再对 180° 取模，为什么：直线没有箭头，1° 与 179° 实际只相差 2°。
    return np.abs((np.asarray(a) - b + 90.0) % 180.0 - 90.0)


def weighted_axial_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    """以线段长度为权重计算轴向圆均值，避免角度跨越 0°/180°时平均错误。"""

    # 把轴向角加倍后转为普通圆周角，为什么：1° 和 179° 不能直接算术平均成 90°。
    doubled = np.deg2rad(2.0 * angles)
    x_value = np.sum(weights * np.cos(doubled))
    y_value = np.sum(weights * np.sin(doubled))
    # atan2 恢复加权方向后再除以 2，为什么：前面为处理轴向等价关系把角度加倍了。
    return float((0.5 * np.rad2deg(math.atan2(y_value, x_value))) % 180.0)


def estimate_hough_angle(mask: np.ndarray, tolerance: float) -> tuple[float, int]:
    """从概率 Hough 候选线段中，以长度加权方式估计占优种植行轴向。"""

    height, width = mask.shape
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    min_length = max(35, int(0.06 * max(height, width)))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720.0,
        threshold=max(30, int(0.018 * max(height, width))),
        minLineLength=min_length,
        maxLineGap=max(20, int(0.025 * max(height, width))),
    )
    # 候选线过少时主动失败，为什么：随意指定方向会输出外观合理但位置错误的行线。
    if lines is None or len(lines) < 3:
        raise RuntimeError("Hough 候选线不足，无法可靠估计种植行方向")

    angles: list[float] = []
    lengths: list[float] = []
    # 遍历 Hough 有限线段并提取角度与长度，为什么：长线比短叶缘更能代表种植行方向。
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = math.hypot(dx, dy)
        # 再次过滤短线，为什么：参数取整后仍可能混入方向不稳定的短边缘。
        if length >= min_length:
            angles.append(math.degrees(math.atan2(dy, dx)) % 180.0)
            lengths.append(length)

    angle_array = np.asarray(angles, dtype=np.float64)
    weight_array = np.asarray(lengths, dtype=np.float64)
    initial = weighted_axial_mean(angle_array, weight_array)
    keep = axial_difference_deg(angle_array, initial) <= tolerance
    # 主方向簇至少需要三条线，为什么：一两条局部叶缘不足以证明整幅图存在稳定行向。
    if int(np.count_nonzero(keep)) < 3:
        raise RuntimeError("Hough 方向候选未形成稳定主簇")
    refined = weighted_axial_mean(angle_array[keep], weight_array[keep])
    return refined, int(np.count_nonzero(keep))


def estimate_spacing(profile: np.ndarray) -> int:
    """用横向投影的一维自相关估计像素行距，避免依赖航高或 GSD。"""

    centered = profile - np.mean(profile)
    correlation = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    min_lag = max(12, int(0.012 * len(profile)))
    max_lag = min(len(profile) // 3, int(0.25 * len(profile)))
    # 合理滞后区间不存在时终止，为什么：过窄投影无法形成可信周期。
    if max_lag <= min_lag:
        raise RuntimeError("横向投影宽度不足，无法估计像素行距")
    search = correlation[min_lag:max_lag]
    peaks, _ = find_peaks(search, distance=min_lag)
    # 优先选自相关局部峰，为什么：周期位置应是局部相似度极大值。
    if len(peaks) > 0:
        best = int(peaks[np.argmax(search[peaks])])
    else:
        # 无局部峰时使用区间最大值，为什么：为弱周期图像提供保守回退而不是直接崩溃。
        best = int(np.argmax(search))
    return min_lag + best


def sample_row_points(mask: np.ndarray, center_x: int, half_width: int) -> np.ndarray:
    """在行走廊中按纵向分箱提取中心代表点，降低大叶片面积偏置。"""

    height, width = mask.shape
    x0, x1 = max(0, center_x - half_width), min(width, center_x + half_width + 1)
    bin_size = max(8, height // 120)
    points: list[tuple[float, float]] = []
    # 沿纵向分箱采样，为什么：每段行程最多贡献一点，使拟合权重在纵向近似均匀。
    for y0 in range(0, height, bin_size):
        y1 = min(height, y0 + bin_size)
        ys, xs = np.nonzero(mask[y0:y1, x0:x1])
        # 每箱至少需要四个植被像素，为什么：防止单个杂草点成为行中心证据。
        if len(xs) >= 4:
            points.append((float(np.median(xs + x0)), float(np.median(ys + y0))))
    return np.asarray(points, dtype=np.float32)


def build_enhanced_mask(image: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """生成强植被和弱苗增强掩膜；弱响应还必须满足绿色通道优势。"""

    strong = build_hsv_mask(image, args)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(image)
    green_i = green.astype(np.int16)
    red_i = red.astype(np.int16)
    blue_i = blue.astype(np.int16)
    # 使用有符号整数计算 ExG，为什么：uint8 做减法可能发生下溢回绕并产生错误大值。
    excess_green = 2 * green_i - red_i - blue_i

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
    # 遍历弱绿色连通域，为什么：弱苗门限可小于强掩膜，但仍需删除单像素色噪声。
    for label in range(1, count):
        # 至少保留六个像素，为什么：兼顾小苗召回率与噪声抑制。
        if stats[label, cv2.CC_STAT_AREA] >= 6:
            weak_clean[labels == label] = 255
    strong_corridor = cv2.dilate(strong, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    weak_near_rows = cv2.bitwise_and(weak_clean, strong_corridor)
    enhanced = cv2.bitwise_or(strong, weak_near_rows)
    return strong, enhanced


def corrected_rotation_matrix(shape: tuple[int, int], angle_deg: float) -> tuple[np.ndarray, tuple[int, int]]:
    """构造把行轴旋到竖直方向且不裁切画面的仿射矩阵。"""

    height, width = shape
    center = (width / 2.0, height / 2.0)
    rotation_deg = angle_deg - 90.0
    matrix = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(height * sine + width * cosine)
    new_height = int(height * cosine + width * sine)
    # 调整平移分量，为什么：只扩大画布而不平移会使旋转内容偏移并被边界裁掉。
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    return matrix, (new_width, new_height)


def projection_periodicity_score(mask: np.ndarray, angle_deg: float) -> float:
    """衡量给定方向法向上的植被密度起伏，用于区分真实行向和叶缘伪方向。"""

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
    # 有效投影过短或几乎无植被时返回零，为什么：窄斜角条带会产生虚假高方差。
    if len(density) < 20 or float(np.mean(density)) <= 1e-8:
        return 0.0
    return float(np.std(density) / np.mean(density))


def detect_direction_regions(strong_mask: np.ndarray, global_angle: float) -> list[tuple[int, int, float]]:
    """检测一次上下正交方向切换；无稳定切换时整图沿全局方向处理。"""

    height = strong_mask.shape[0]
    strip_count = 12
    orthogonal_angle = (global_angle + 90.0) % 180.0
    orthogonal_votes: list[bool] = []
    # 比较十二个水平条带的两种周期得分，为什么：局部 Hough 容易受短叶缘干扰。
    for strip_index in range(strip_count):
        y0 = int(strip_index * height / strip_count)
        y1 = int((strip_index + 1) * height / strip_count)
        strip = strong_mask[y0:y1]
        global_score = projection_periodicity_score(strip, global_angle)
        orthogonal_score = projection_periodicity_score(strip, orthogonal_angle)
        orthogonal_votes.append(orthogonal_score > 1.15 * max(global_score, 1e-8))

    switch_index: int | None = None
    # 搜索连续两票，为什么：单个异常条带可能只是地头、缺苗或局部杂草。
    for index in range(strip_count - 1):
        # 找到首次稳定切换后停止，为什么：当前算法明确只支持一次上下区域转换。
        if orthogonal_votes[index] and orthogonal_votes[index + 1]:
            switch_index = index
            break
    # 没有稳定正交区域时返回单区域，为什么：避免对单方向地块进行多余切割。
    if switch_index is None:
        return [(0, height, global_angle)]

    boundary = int((switch_index + 0.5) * height / strip_count)
    return [(0, boundary, global_angle), (boundary, height, orthogonal_angle)]


def detect_rows_in_region(
    mask: np.ndarray,
    angle_deg: float,
    y_offset: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray, int, float]], int]:
    """在单方向区域中通过投影峰定位种植行，并拟合由植被支持限定的线段。"""

    matrix, output_size = corrected_rotation_matrix(mask.shape, angle_deg)
    rotated = cv2.warpAffine(mask, matrix, output_size, flags=cv2.INTER_NEAREST, borderValue=0)
    raw_profile = np.sum(rotated > 0, axis=0).astype(np.float64)
    smooth_profile = gaussian_filter1d(raw_profile, sigma=max(1.5, 0.002 * len(raw_profile)))
    spacing = estimate_spacing(smooth_profile)
    prominence = max(float(np.max(smooth_profile)) * 0.06, float(np.std(smooth_profile)) * 0.30)
    peaks, _ = find_peaks(smooth_profile, distance=max(8, int(0.55 * spacing)), prominence=prominence)

    inverse = cv2.invertAffineTransform(matrix)
    segments: list[tuple[np.ndarray, np.ndarray, int, float]] = []
    half_width = max(4, int(0.28 * spacing))
    # 每个投影峰只拟合一条中心线，为什么：从机制上减少把同一行两侧叶缘重复报线。
    for peak in peaks:
        points = sample_row_points(rotated, int(peak), half_width)
        # 支持点或跨度不足时跳过，为什么：短斑更可能是杂草而不是完整种植行。
        if len(points) < 12 or np.ptp(points[:, 1]) < 0.18 * rotated.shape[0]:
            continue
        vx, vy, x0, y0 = (float(value) for value in cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1))
        # 旋正后仍明显倾斜或接近水平时回退，为什么：局部弱苗可能把鲁棒拟合拉偏。
        if abs(vx) > 0.25 or abs(vy) < 1e-6:
            vx, vy, x0 = 0.0, 1.0, float(np.median(points[:, 0]))
        y_start, y_end = np.quantile(points[:, 1], [0.02, 0.98])
        start_rotated = np.array([x0 + vx / vy * (y_start - y0), y_start, 1.0], dtype=np.float64)
        end_rotated = np.array([x0 + vx / vy * (y_end - y0), y_end, 1.0], dtype=np.float64)
        start = inverse @ start_rotated
        end = inverse @ end_rotated
        start[1] += y_offset
        end[1] += y_offset
        coverage = float(np.ptp(points[:, 1]) / max(1, rotated.shape[0]))
        segments.append((start, end, len(points), coverage))
    return segments, spacing


def draw_and_serialize(
    original: np.ndarray,
    segments: list[tuple[np.ndarray, np.ndarray, int, float]],
    work_scale: float,
) -> tuple[np.ndarray, list[RowSegment]]:
    """把工作坐标线段映射回原图，绘制橙线，并生成 JSON 可序列化记录。"""

    canvas = original.copy()
    orange = (0, 165, 255)
    line_width = max(4, int(round(5 / work_scale)))
    radius = max(7, int(round(9 / work_scale)))
    rows: list[RowSegment] = []
    height, width = original.shape[:2]
    # 逐条映射、裁剪和过滤，为什么：旋转画布的浮点端点可能超出原图边界。
    for row_id, (start_work, end_work, support, coverage) in enumerate(segments, start=1):
        start = np.rint(start_work / work_scale).astype(int)
        end = np.rint(end_work / work_scale).astype(int)
        start[0], start[1] = np.clip(start[0], 0, width - 1), np.clip(start[1], 0, height - 1)
        end[0], end[1] = np.clip(end[0], 0, width - 1), np.clip(end[1], 0, height - 1)
        length = float(np.linalg.norm(end - start))
        # 映射后线段过短则删除，为什么：短线没有稳定的田间行方向意义。
        if length < 0.12 * max(height, width):
            continue
        angle = float(math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 180.0)
        p1, p2 = tuple(start.tolist()), tuple(end.tolist())
        cv2.line(canvas, p1, p2, orange, line_width, cv2.LINE_AA)
        cv2.circle(canvas, p1, radius, orange, -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, radius, orange, -1, cv2.LINE_AA)
        rows.append(RowSegment(row_id, list(p1), list(p2), angle, length, support, coverage))
    return canvas, rows


def write_image_checked(path: Path, image: np.ndarray) -> None:
    """写出图片并检查 OpenCV 返回值，避免磁盘失败时仍记录为成功。"""

    # imwrite 用布尔值报告成功与否，为什么：部分路径或磁盘错误并不一定直接抛出异常。
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"无法写出图像：{path}")


def process_image(path: Path, output_dir: Path, args: argparse.Namespace) -> dict:
    """执行单张图片的完整检测流程并输出掩膜、叠加图和结构化 JSON。"""

    original = cv2.imread(str(path), cv2.IMREAD_COLOR)
    # 解码失败时立即终止当前图片，为什么：不能把空输入误记为成功结果。
    if original is None:
        raise RuntimeError(f"无法读取图像：{path}")
    work, scale = resize_for_work(original, args.max_side)
    strong_mask, enhanced_mask = build_enhanced_mask(work, args)
    mask_ratio = float(np.mean(enhanced_mask > 0))
    # 掩膜比例异常时拒绝继续，为什么：阈值失控后强行出线会产生高置信度外观的错误结果。
    if not 0.002 <= mask_ratio <= 0.65:
        raise RuntimeError(f"增强掩膜占比异常：{mask_ratio:.4f}")

    global_angle, hough_support = estimate_hough_angle(strong_mask, args.angle_tolerance)
    regions = detect_direction_regions(strong_mask, global_angle)
    all_segments: list[tuple[np.ndarray, np.ndarray, int, float]] = []
    region_records: list[dict] = []
    # 各方向区域独立检测，为什么：异方向区域不应让一条直线错误贯穿整幅图。
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

    overlay, rows = draw_and_serialize(original, all_segments, scale)
    # 完全没有最终行时视为失败，为什么：空白输出不能代表算法处理成功。
    if not rows:
        raise RuntimeError("未获得满足支持度和长度要求的种植行")
    mask_full = cv2.resize(enhanced_mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_path = output_dir / f"{path.stem}_mask_latest.png"
    overlay_path = output_dir / f"{path.stem}_rows_latest.png"
    json_path = output_dir / f"{path.stem}_rows_latest.json"
    write_image_checked(mask_path, mask_full)
    write_image_checked(overlay_path, overlay)
    result = {
        "version": "latest-standalone",
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
        "outputs": {"mask": str(mask_path), "overlay": str(overlay_path), "json": str(json_path)},
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    """依次处理显式指定的图片；单图失败不会中断同批次的其他图片。"""

    configure_console_encoding()
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    images = collect_images(input_path, args.names)
    # 按收集后的稳定顺序逐张处理，为什么：固定顺序便于重复实验和对照结果。
    for path in images:
        name = path.name
        try:
            result = process_image(path, output_dir, args)
            summary.append({"source": str(path), "status": "success", "row_count": result["row_count"]})
            print(f"[最新版成功] {name}: {result['row_count']} 行，{result['direction_region_count']} 个方向区域")
        except Exception as exc:
            # 捕获单图异常，为什么：批处理中一张坏图不应阻止其他图片继续运行。
            summary.append({"source": str(path), "status": "failed", "error": str(exc)})
            print(f"[最新版失败] {name}: {exc}")
    (output_dir / "run_summary_latest.json").write_text(
        json.dumps({"parameters": vars(args), "images": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# 只有直接执行脚本时运行主函数，为什么：被其他工具导入时不应自动读写图片。
if __name__ == "__main__":
    main()
