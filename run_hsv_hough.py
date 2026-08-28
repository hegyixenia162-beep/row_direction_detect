"""使用 HSV 植被分割与 Hough 主方向估计检测棉花种植行。"""

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


@dataclass
class RowSegment:
    """保存一条种植行在原图坐标系中的有限线段及其质量特征。"""

    row_id: int
    start: list[int]
    end: list[int]
    angle_deg: float
    length_px: float
    support_points: int
    longitudinal_coverage: float


def parse_args() -> argparse.Namespace:
    """解析命令行参数；把阈值显式暴露出来，便于针对不同批次航片复现实验。"""

    parser = argparse.ArgumentParser(description="HSV + Hough 棉花种植行检测")
    parser.add_argument("--input", default="test_photo", help="输入图像或图像目录")
    parser.add_argument("--output", default="outputs", help="结果输出目录")
    parser.add_argument("--max-side", type=int, default=1800, help="工作图像最长边")
    parser.add_argument("--h-min", type=int, default=25, help="OpenCV HSV 色相下限")
    parser.add_argument("--h-max", type=int, default=100, help="OpenCV HSV 色相上限")
    parser.add_argument("--s-min", type=int, default=35, help="饱和度下限")
    parser.add_argument("--v-min", type=int, default=25, help="亮度下限")
    parser.add_argument("--min-component", type=int, default=18, help="工作尺度最小连通域面积")
    parser.add_argument("--angle-tolerance", type=float, default=12.0, help="Hough 候选角容差")
    return parser.parse_args()


def list_images(input_path: Path) -> list[Path]:
    """收集支持的图像；目录模式按文件名排序，保证多次运行顺序一致。"""

    # 如果输入是单个文件，只处理该文件，避免误扫同目录中的其他数据。
    if input_path.is_file():
        return [input_path]

    suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    # 遍历目录并限制扩展名，避免把已有输出或非图像文件送入 OpenCV。
    return sorted(path for path in input_path.iterdir() if path.suffix.lower() in suffixes)


def resize_for_work(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """等比例缩小大图；返回工作图和工作坐标相对原图的缩放比。"""

    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    # 仅在原图超过工作尺寸时缩小，避免对小图无意义地放大并制造插值纹理。
    if scale < 1.0:
        work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        work = image.copy()
    return work, scale


def build_hsv_mask(image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """用 HSV 阈值提取植被，并通过形态学和面积过滤抑制背景噪声。"""

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
    # 从标签 1 开始遍历，因为标签 0 是背景；面积门限用于删除零散彩色噪声。
    for label in range(1, count):
        # 只有达到最小苗体候选面积的连通域才写回掩膜。
        if stats[label, cv2.CC_STAT_AREA] >= args.min_component:
            cleaned[labels == label] = 255
    return cleaned


def axial_difference_deg(a: np.ndarray | float, b: float) -> np.ndarray:
    """计算无方向直线的最小夹角；0° 与 180° 被视为同一轴向。"""

    return np.abs((np.asarray(a) - b + 90.0) % 180.0 - 90.0)


def weighted_axial_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    """用二倍角圆均值估计直线轴向，避免 1° 与 179° 被错误平均成 90°。"""

    doubled = np.deg2rad(2.0 * angles)
    x_value = np.sum(weights * np.cos(doubled))
    y_value = np.sum(weights * np.sin(doubled))
    return float((0.5 * np.rad2deg(math.atan2(y_value, x_value))) % 180.0)


def estimate_hough_angle(mask: np.ndarray, tolerance: float) -> tuple[float, int]:
    """从植被掩膜边缘的概率 Hough 线段中估计占优种植行轴向。"""

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
    # 没有足够长的 Hough 线段时主动失败，因为随意给定角度会产生貌似合理的错误行线。
    if lines is None or len(lines) < 3:
        raise RuntimeError("Hough 候选线不足，无法可靠估计种植行方向")

    angles: list[float] = []
    lengths: list[float] = []
    # 遍历概率 Hough 返回的有限线段，并用长度作为方向投票权重。
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = math.hypot(dx, dy)
        # 再次检查长度是为了防止 OpenCV 参数取整后混入不稳定短边缘。
        if length >= min_length:
            angles.append(math.degrees(math.atan2(dy, dx)) % 180.0)
            lengths.append(length)

    angle_array = np.asarray(angles, dtype=np.float64)
    weight_array = np.asarray(lengths, dtype=np.float64)
    initial = weighted_axial_mean(angle_array, weight_array)
    keep = axial_difference_deg(angle_array, initial) <= tolerance
    # 主簇过小时说明画面存在多方向或背景直线主导，此时不应伪造单一方向。
    if int(np.count_nonzero(keep)) < 3:
        raise RuntimeError("Hough 方向候选未形成稳定主簇")
    refined = weighted_axial_mean(angle_array[keep], weight_array[keep])
    return refined, int(np.count_nonzero(keep))


def rotation_matrix_to_vertical(shape: tuple[int, int], angle_deg: float) -> tuple[np.ndarray, tuple[int, int]]:
    """构造把主方向旋转到竖直方向且完整保留画面的仿射矩阵。"""

    height, width = shape
    rotation_deg = 90.0 - angle_deg
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(height * sine + width * cosine)
    new_height = int(height * cosine + width * sine)
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    return matrix, (new_width, new_height)


def estimate_spacing(profile: np.ndarray) -> int:
    """通过去均值投影的一维自相关估计行距，避免依赖飞行高度或 GSD。"""

    centered = profile - np.mean(profile)
    correlation = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    min_lag = max(12, int(0.012 * len(profile)))
    max_lag = min(len(profile) // 3, int(0.25 * len(profile)))
    # 搜索区间无效意味着有效横向宽度太小，不能稳定估计周期行距。
    if max_lag <= min_lag:
        raise RuntimeError("横向投影宽度不足，无法估计像素行距")
    search = correlation[min_lag:max_lag]
    peaks, _ = find_peaks(search, distance=min_lag)
    # 优先选自相关局部峰；没有局部峰时使用区间最大值作为保守回退。
    if len(peaks) > 0:
        best = int(peaks[np.argmax(search[peaks])])
    else:
        best = int(np.argmax(search))
    return min_lag + best


def sample_row_points(mask: np.ndarray, center_x: int, half_width: int) -> np.ndarray:
    """在行走廊内按纵向分箱取横坐标中位数，降低大叶片面积对中心拟合的偏置。"""

    height, width = mask.shape
    x0, x1 = max(0, center_x - half_width), min(width, center_x + half_width + 1)
    bin_size = max(8, height // 120)
    points: list[tuple[float, float]] = []
    # 按纵向小区间采样，使每一段行程最多贡献一个代表点并获得近似均匀权重。
    for y0 in range(0, height, bin_size):
        y1 = min(height, y0 + bin_size)
        ys, xs = np.nonzero(mask[y0:y1, x0:x1])
        # 小区间至少需要若干植被像素，避免单个杂草点成为行中心证据。
        if len(xs) >= 4:
            points.append((float(np.median(xs + x0)), float(np.median(ys + y0))))
    return np.asarray(points, dtype=np.float32)


def detect_rows(mask: np.ndarray, angle_deg: float) -> tuple[list[tuple[np.ndarray, np.ndarray, int, float]], int, np.ndarray]:
    """在旋正掩膜上定位每一行，并生成由真实植被支持限定的有限中心线段。"""

    matrix, output_size = rotation_matrix_to_vertical(mask.shape, angle_deg)
    rotated = cv2.warpAffine(mask, matrix, output_size, flags=cv2.INTER_NEAREST, borderValue=0)
    raw_profile = np.sum(rotated > 0, axis=0).astype(np.float64)
    smooth_profile = gaussian_filter1d(raw_profile, sigma=max(2.0, 0.004 * len(raw_profile)))
    spacing = estimate_spacing(smooth_profile)
    prominence = max(np.max(smooth_profile) * 0.08, np.std(smooth_profile) * 0.5)
    # 自相关可能把隔一行的强周期识别为基频；较小的抑制距离先保留相邻弱行，后续再靠长程支持过滤假峰。
    peaks, properties = find_peaks(smooth_profile, distance=max(8, int(0.22 * spacing)), prominence=prominence)

    inverse = cv2.invertAffineTransform(matrix)
    segments: list[tuple[np.ndarray, np.ndarray, int, float]] = []
    half_width = max(5, int(0.30 * spacing))
    # 逐峰处理可保证每个法向位置至多产生一条中心线，从机制上抑制双叶缘线。
    for peak_index, peak in enumerate(peaks):
        points = sample_row_points(rotated, int(peak), half_width)
        # 代表点过少或纵向跨度太短时，候选更可能是杂草斑而不是完整种植行。
        if len(points) < 12 or np.ptp(points[:, 1]) < 0.18 * rotated.shape[0]:
            continue

        vx, vy, x0, y0 = (float(v) for v in cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1))
        # 旋正后行线应接近竖直；过大的横向斜率说明该峰被异向杂草或边缘污染。
        if abs(vx) > 0.32:
            vx, vy, x0 = 0.0, 1.0, float(np.median(points[:, 0]))
        y_start, y_end = np.quantile(points[:, 1], [0.02, 0.98])
        start_rot = np.array([x0 + vx / vy * (y_start - y0), y_start, 1.0], dtype=np.float64)
        end_rot = np.array([x0 + vx / vy * (y_end - y0), y_end, 1.0], dtype=np.float64)
        start = inverse @ start_rot
        end = inverse @ end_rot
        coverage = float(np.ptp(points[:, 1]) / max(1, rotated.shape[0]))
        segments.append((start, end, len(points), coverage))
    return segments, spacing, rotated


def draw_and_serialize(
    original: np.ndarray,
    segments: list[tuple[np.ndarray, np.ndarray, int, float]],
    work_scale: float,
) -> tuple[np.ndarray, list[RowSegment]]:
    """将工作坐标线段映射回原图，绘制橙线和端点，并生成 JSON 可序列化记录。"""

    canvas = original.copy()
    orange = (0, 165, 255)
    line_width = max(4, int(round(5 / work_scale)))
    radius = max(7, int(round(9 / work_scale)))
    rows: list[RowSegment] = []
    height, width = original.shape[:2]
    # 逐条映射和裁剪端点，防止旋转画布边缘的浮点坐标越出原图。
    for row_id, (start_work, end_work, support, coverage) in enumerate(segments, start=1):
        start = np.rint(start_work / work_scale).astype(int)
        end = np.rint(end_work / work_scale).astype(int)
        start[0], start[1] = np.clip(start[0], 0, width - 1), np.clip(start[1], 0, height - 1)
        end[0], end[1] = np.clip(end[0], 0, width - 1), np.clip(end[1], 0, height - 1)
        length = float(np.linalg.norm(end - start))
        # 映射后过短的线没有稳定方向意义，因此不写入最终图和 JSON。
        if length < 0.12 * max(height, width):
            continue
        angle = float(math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 180.0)
        p1, p2 = tuple(start.tolist()), tuple(end.tolist())
        cv2.line(canvas, p1, p2, orange, line_width, cv2.LINE_AA)
        cv2.circle(canvas, p1, radius, orange, -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, radius, orange, -1, cv2.LINE_AA)
        rows.append(RowSegment(row_id, list(p1), list(p2), angle, length, support, coverage))
    return canvas, rows


def process_image(path: Path, output_dir: Path, args: argparse.Namespace) -> dict:
    """执行单张航片的完整流程，并把掩膜、呈现图和结构化结果写入磁盘。"""

    original = cv2.imread(str(path), cv2.IMREAD_COLOR)
    # 解码失败时立即给出带文件名的明确错误，方便批处理日志定位损坏文件。
    if original is None:
        raise RuntimeError(f"无法读取图像：{path}")
    work, scale = resize_for_work(original, args.max_side)
    mask = build_hsv_mask(work, args)
    mask_ratio = float(np.mean(mask > 0))
    # 掩膜太少通常代表阈值漏掉植被，太多则代表背景被误分；二者均不宜继续强行出线。
    if not 0.002 <= mask_ratio <= 0.55:
        raise RuntimeError(f"HSV 掩膜占比异常：{mask_ratio:.4f}")

    angle, hough_support = estimate_hough_angle(mask, args.angle_tolerance)
    segments, spacing_work, _ = detect_rows(mask, angle)
    overlay, rows = draw_and_serialize(original, segments, scale)
    # 没有任何通过长度和支持度筛选的行时，将该图标记为失败而不是输出空白成功结果。
    if not rows:
        raise RuntimeError("未获得满足支持度和长度要求的有限种植行")

    mask_full = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_path = output_dir / f"{path.stem}_mask.png"
    overlay_path = output_dir / f"{path.stem}_rows.png"
    json_path = output_dir / f"{path.stem}_rows.json"
    cv2.imwrite(str(mask_path), mask_full)
    cv2.imwrite(str(overlay_path), overlay)
    result = {
        "source": str(path),
        "image_size": {"width": original.shape[1], "height": original.shape[0]},
        "work_scale": scale,
        "hsv_mask_ratio": mask_ratio,
        "dominant_row_angle_deg": angle,
        "hough_support_lines": hough_support,
        "estimated_row_spacing_px": spacing_work / scale,
        "row_count": len(rows),
        "rows": [asdict(row) for row in rows],
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    """批量处理输入并记录每张图的成功或失败状态，保证局部失败不终止整批任务。"""

    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(input_path)
    # 输入为空时停止运行，避免生成一个看似成功但没有处理对象的汇总文件。
    if not images:
        raise SystemExit(f"未找到输入图像：{input_path}")

    summary: list[dict] = []
    # 对每张图独立捕获异常，使阈值不适合某一张图时仍可观察其他图的效果。
    for image_path in images:
        try:
            result = process_image(image_path, output_dir, args)
            summary.append({"source": str(image_path), "status": "success", "row_count": result["row_count"]})
            print(f"[成功] {image_path.name}: {result['row_count']} 行")
        except Exception as exc:
            summary.append({"source": str(image_path), "status": "failed", "error": str(exc)})
            print(f"[失败] {image_path.name}: {exc}")

    run_record = {"parameters": vars(args), "images": summary}
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# 仅在脚本直接执行时启动批处理，导入模块做单元测试时不产生文件副作用。
if __name__ == "__main__":
    main()
