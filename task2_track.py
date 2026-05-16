import os
import cv2
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from collections import defaultdict


def run_tracking(
    video_path: str,
    model_path: str,
    output_path: str = "tracked_output.mp4",
    line_y: int = None,
    line_x1: int = None,
    line_x2: int = None,
):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if line_y is None:
        line_y = height // 2
    if line_x1 is None:
        line_x1 = 0
    if line_x2 is None:
        line_x2 = width

    prev_centers: dict[int, float] = {}
    total_crossed: set[int] = set()

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            iou=0.5,
            verbose=False,
        )

        annotated = results[0].plot()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()   # [cx, cy, w, h]
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, tid in zip(boxes, track_ids):
                cx, cy = float(box[0]), float(box[1])
                tid = int(tid)

                if tid in prev_centers:
                    prev_cy = prev_centers[tid]

                    if (prev_cy < line_y <= cy) or (cy < line_y <= prev_cy):
                        if tid not in total_crossed:
                            total_crossed.add(tid)

                prev_centers[tid] = cy

        cv2.line(annotated, (line_x1, line_y), (line_x2, line_y), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"Crossed: {len(total_crossed)}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
        )

        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"跟踪完成, 共 {frame_idx} 帧, 越线物体总数: {len(total_crossed)}")
    print(f"输出视频: {output_path}")


def annotate_spread_ids(frame, results, conf=0.25):
    annotated = frame.copy()
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return annotated

    xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else None
    cls_ids = boxes.cls.int().cpu().numpy() if boxes.cls is not None else None
    track_ids = boxes.id.int().cpu().numpy() if boxes.id is not None else None
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None

    if xyxy is None:
        return annotated

    class_names = results[0].names  # {0: "pedestrian", ...}

    for idx, bbox in enumerate(xyxy):
        x1, y1, x2, y2 = map(int, bbox)
        tid = int(track_ids[idx]) if track_ids is not None else -1
        cls_id = int(cls_ids[idx]) if cls_ids is not None else -1

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)

        conf_str = f"{confs[idx]:.2f}" if confs is not None else ""
        if tid >= 0:
            label = f"ID:{tid} {conf_str}"
        elif cls_id >= 0:
            label = f"{class_names.get(cls_id, f'cls{cls_id}')} {conf_str}"
        else:
            label = ""

        if not label:
            continue

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        if tid >= 0:
            if tid % 2 == 0:
                offset_x = (tid * 7) % 15 - 7  # -7 ~ +7 px
                label_x = max(2, x1 + offset_x)
                label_y = max(th + 2, y1 - 4)
            else:
                offset_x = (tid * 11) % 15 - 7
                label_x = max(2, x1 + offset_x)
                label_y = min(annotated.shape[0] - 4, y2 + th + 4)
        else:
            label_x = max(2, x1)
            label_y = max(th + 2, y1 - 4)

        label_x = min(label_x, annotated.shape[1] - tw - 2)

        cv2.rectangle(
            annotated,
            (label_x - 2, label_y - th - 2),
            (label_x + tw + 2, label_y + 2),
            (0, 0, 0),
            -1,
        )

        color = (0, 255, 255) if (tid >= 0 and tid % 2 == 0) else (255, 200, 0)
        cv2.putText(annotated, label, (label_x, label_y), font, font_scale, color, thickness)

    return annotated


def extract_keyframes(
    video_path: str,
    model_path: str,
    start_frame: int,
    num_frames: int = 4,
    output_dir: str = "keyframes",
):
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            iou=0.5,
            verbose=False,
        )

        annotated = annotate_spread_ids(frame, results)

        cv2.putText(
            annotated,
            f"Frame {start_frame + i}",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        out_path = os.path.join(output_dir, f"frame_{start_frame + i:04d}.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"保存: {out_path}")

    cap.release()
    print(f"关键帧已保存至 {output_dir}/")


if __name__ == "__main__":
    VIDEO_PATH = "track.mp4"
    MODEL_PATH = "runs/detect/visdrone_yolov8/weights/best.pt"
    OUTPUT_VIDEO = "tracked_output.mp4"

    # 计数线位置
    LINE_Y = 500
    LINE_X1 = 0
    LINE_X2 = 1280

    # 遮挡分析关键帧起始位置
    OCCLUSION_START_FRAME = 9

    # 视频跟踪 + 越线计数
    run_tracking(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        output_path=OUTPUT_VIDEO,
        line_y=LINE_Y,
        line_x1=LINE_X1,
        line_x2=LINE_X2,
    )

    # 提取遮挡关键帧
    extract_keyframes(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        start_frame=OCCLUSION_START_FRAME,
        num_frames=4,
        output_dir="keyframes_occlusion",
    )
