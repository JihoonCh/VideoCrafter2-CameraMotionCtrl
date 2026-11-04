import argparse
import csv
from pathlib import Path

import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def get_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for optical-flow-based temporal metrics. "
            "Install it with `pip install opencv-python` or `pip install opencv-contrib-python`."
        ) from exc
    return cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate self-consistency temporal metrics for generated videos."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a video file, a directory of videos, or a directory of extracted frames.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search videos when --input is a directory.",
    )
    parser.add_argument(
        "--flow-method",
        type=str,
        default="farneback",
        choices=["farneback", "dis"],
        help="Optical flow backend.",
    )
    parser.add_argument(
        "--disable-fb-check",
        action="store_true",
        help="Disable forward-backward consistency masking.",
    )
    parser.add_argument(
        "--fb-threshold",
        type=float,
        default=1.0,
        help="Base threshold for forward-backward consistency masking.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV output path.",
    )
    return parser.parse_args()


def load_frames_from_video(video_path):
    cv2 = get_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb.astype(np.float32) / 255.0)

    cap.release()
    if len(frames) < 2:
        raise RuntimeError(f"Video must have at least 2 frames: {video_path}")
    return frames


def load_frames_from_directory(frame_dir):
    cv2 = get_cv2()
    frame_dir = Path(frame_dir)
    frame_paths = sorted(
        path for path in frame_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(frame_paths) < 2:
        raise RuntimeError(f"Frame directory must contain at least 2 images: {frame_dir}")

    frames = []
    for frame_path in frame_paths:
        frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"Could not read frame: {frame_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb.astype(np.float32) / 255.0)
    return frames


def resolve_inputs(input_path, recursive=False):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {input_path}")
        return [input_path]

    has_images = any(path.suffix.lower() in IMAGE_EXTENSIONS for path in input_path.iterdir())
    if has_images:
        return [input_path]

    pattern = "**/*" if recursive else "*"
    video_paths = [
        path for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not video_paths:
        raise RuntimeError(f"No videos found in: {input_path}")
    return sorted(video_paths)


def build_flow_estimator(method):
    cv2 = get_cv2()
    if method == "farneback":
        return None
    if method == "dis":
        return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    raise ValueError(f"Unsupported flow method: {method}")


def to_gray_u8(frame):
    cv2 = get_cv2()
    return cv2.cvtColor((frame * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)


def estimate_flow(source_frame, target_frame, method="farneback", estimator=None):
    cv2 = get_cv2()
    source_gray = to_gray_u8(source_frame)
    target_gray = to_gray_u8(target_frame)

    if method == "farneback":
        return cv2.calcOpticalFlowFarneback(
            source_gray,
            target_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

    return estimator.calc(source_gray, target_gray, None)


def warp_frame(frame, flow):
    cv2 = get_cv2()
    height, width = frame.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)

    warped = cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    valid = (
        (map_x >= 0.0) & (map_x <= width - 1.0) &
        (map_y >= 0.0) & (map_y <= height - 1.0)
    )
    return warped, valid, map_x, map_y


def sample_flow(flow, map_x, map_y):
    cv2 = get_cv2()
    sampled_x = cv2.remap(flow[..., 0], map_x, map_y, interpolation=cv2.INTER_LINEAR)
    sampled_y = cv2.remap(flow[..., 1], map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return np.stack([sampled_x, sampled_y], axis=-1)


def forward_backward_mask(forward_flow, backward_flow, map_x, map_y, base_threshold=1.0):
    sampled_forward = sample_flow(forward_flow, map_x, map_y)
    fb_error = backward_flow + sampled_forward

    error_norm = np.linalg.norm(fb_error, axis=-1)
    backward_norm = np.linalg.norm(backward_flow, axis=-1)
    forward_norm = np.linalg.norm(sampled_forward, axis=-1)

    threshold = base_threshold + 0.01 * (backward_norm + forward_norm)
    return error_norm <= threshold


def masked_mean(values, mask):
    if not np.any(mask):
        return float("nan")
    return float(values[mask].mean())


def evaluate_frame_pair(
    prev_frame,
    curr_frame,
    method="farneback",
    estimator=None,
    use_fb_check=True,
    fb_threshold=1.0,
):
    forward_flow = estimate_flow(prev_frame, curr_frame, method=method, estimator=estimator)
    backward_flow = estimate_flow(curr_frame, prev_frame, method=method, estimator=estimator)

    warped_prev, valid_mask, map_x, map_y = warp_frame(prev_frame, backward_flow)
    mask = valid_mask

    if use_fb_check:
        fb_mask = forward_backward_mask(
            forward_flow=forward_flow,
            backward_flow=backward_flow,
            map_x=map_x,
            map_y=map_y,
            base_threshold=fb_threshold,
        )
        mask = mask & fb_mask

    warped_error_map = np.mean(np.abs(curr_frame - warped_prev), axis=2)
    raw_diff_map = np.mean(np.abs(curr_frame - prev_frame), axis=2)

    warping_error = masked_mean(warped_error_map, mask)
    raw_difference = masked_mean(raw_diff_map, mask)

    if np.isnan(warping_error) or np.isnan(raw_difference):
        temporal_consistency = float("nan")
    else:
        temporal_consistency = 1.0 - (warping_error / (raw_difference + 1e-8))

    return {
        "warping_error": warping_error,
        "temporal_consistency": temporal_consistency,
        "valid_ratio": float(mask.mean()),
    }


def load_sequence(input_path):
    path = Path(input_path)
    if path.is_dir():
        return load_frames_from_directory(path)
    return load_frames_from_video(path)


def evaluate_sequence(
    input_path,
    method="farneback",
    use_fb_check=True,
    fb_threshold=1.0,
):
    frames = load_sequence(input_path)
    estimator = build_flow_estimator(method)

    pair_metrics = []
    for index in range(1, len(frames)):
        metrics = evaluate_frame_pair(
            prev_frame=frames[index - 1],
            curr_frame=frames[index],
            method=method,
            estimator=estimator,
            use_fb_check=use_fb_check,
            fb_threshold=fb_threshold,
        )
        pair_metrics.append(metrics)

    warping_errors = [m["warping_error"] for m in pair_metrics if not np.isnan(m["warping_error"])]
    temporal_scores = [m["temporal_consistency"] for m in pair_metrics if not np.isnan(m["temporal_consistency"])]
    valid_ratios = [m["valid_ratio"] for m in pair_metrics]

    return {
        "path": str(input_path),
        "num_frames": len(frames),
        "num_pairs": len(pair_metrics),
        "warping_error_mean": float(np.mean(warping_errors)) if warping_errors else float("nan"),
        "temporal_consistency_mean": float(np.mean(temporal_scores)) if temporal_scores else float("nan"),
        "valid_ratio_mean": float(np.mean(valid_ratios)) if valid_ratios else float("nan"),
    }


def write_csv(csv_path, rows):
    fieldnames = [
        "path",
        "num_frames",
        "num_pairs",
        "warping_error_mean",
        "temporal_consistency_mean",
        "valid_ratio_mean",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    input_paths = resolve_inputs(args.input, recursive=args.recursive)

    rows = []
    for input_path in input_paths:
        result = evaluate_sequence(
            input_path=input_path,
            method=args.flow_method,
            use_fb_check=not args.disable_fb_check,
            fb_threshold=args.fb_threshold,
        )
        rows.append(result)
        print(
            f"{result['path']} | frames={result['num_frames']} | "
            f"WE={result['warping_error_mean']:.6f} | "
            f"TC={result['temporal_consistency_mean']:.6f} | "
            f"valid={result['valid_ratio_mean']:.3f}"
        )

    if len(rows) > 1:
        warping_errors = [row["warping_error_mean"] for row in rows if not np.isnan(row["warping_error_mean"])]
        temporal_scores = [row["temporal_consistency_mean"] for row in rows if not np.isnan(row["temporal_consistency_mean"])]
        valid_ratios = [row["valid_ratio_mean"] for row in rows if not np.isnan(row["valid_ratio_mean"])]

        print("\nDataset average")
        print(f"WE={np.mean(warping_errors):.6f}")
        print(f"TC={np.mean(temporal_scores):.6f}")
        print(f"valid={np.mean(valid_ratios):.3f}")

    if args.csv is not None:
        write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
