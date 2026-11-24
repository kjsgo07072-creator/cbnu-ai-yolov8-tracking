"""학습한 YOLOv8 모델로 동영상 객체 추적 + 궤적(trajectory) 저장.

Ultralytics object_tracking 예제(매 프레임 박스 중심점을 이어 그리는 방식)를 바탕으로,
예제가 화면에만 그리고 버리는 궤적을 파일로 따로 저장하도록 고쳤다.

python scripts/track.py --source videos/input_dogs_walk_10s.mp4 --start 0 --duration 5

결과 (outputs/<name>/):
    tracking.mp4        박스 + 클래스 + ID + 궤적 꼬리
    trajectories.csv    frame, time_sec, track_id, class, conf, cx, cy, x1, y1, x2, y2
    trajectories.png    ID별 전체 이동 경로 그림
    summary.json        ID별 등장 프레임 수, 이동 거리 등
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
COLORS = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
          (10, 249, 72), (23, 204, 146), (134, 219, 61), (211, 188, 0), (209, 85, 0)]


def color(tid):
    return COLORS[int(tid) % len(COLORS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", default=str(ROOT / "01_trained_model/best.pt"))
    ap.add_argument("--tracker", default="botsort.yaml", help="botsort.yaml 또는 bytetrack.yaml")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--agnostic-nms", action="store_true", help="클래스가 달라도 겹치는 박스는 하나만 남김 (같은 강아지를 dog·cat 박스로 중복 감지하는 문제 완화)")
    ap.add_argument("--start", type=float, default=0.0, help="시작 시각(초)")
    ap.add_argument("--duration", type=float, default=5.0, help="저장할 길이(초)")
    ap.add_argument("--tail", type=int, default=60, help="영상에 그릴 궤적 꼬리 길이(프레임)")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    src = Path(args.source)
    out = ROOT / "outputs" / (args.name or src.stem)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    names = model.names
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(3)), int(cap.get(4))
    first = int(round(args.start * fps))
    last = first + int(round(args.duration * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    writer = cv2.VideoWriter(str(out / "tracking_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    history = defaultdict(list)     # track_id -> [(cx, cy), ...]  전체 궤적 (예제처럼 자르지 않음)
    track_cls = {}
    last_seen = {}
    rows = []
    background = None

    processed = 0
    for fidx in range(first, last):
        ok, frame = cap.read()
        if not ok:
            break
        processed += 1
        if background is None:
            background = frame.copy()
        res = model.track(frame, persist=True, tracker=args.tracker, conf=args.conf, agnostic_nms=args.agnostic_nms, verbose=False)[0]
        vis = frame.copy()
        b = res.boxes
        if b.id is not None:
            for (x1, y1, x2, y2), tid, c, s in zip(b.xyxy.tolist(), b.id.int().tolist(), b.cls.int().tolist(), b.conf.tolist()):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                history[tid].append((cx, cy))
                last_seen[tid] = fidx
                track_cls[tid] = names[c]
                rows.append([fidx - first, round((fidx - first) / fps, 3), tid, names[c], round(s, 3),
                             round(cx, 1), round(cy, 1), round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
                col = color(tid)
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 3)
                label = f"{names[c]} #{tid} {s:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(vis, (int(x1), int(y1) - th - 10), (int(x1) + tw + 6, int(y1)), col, -1)
                cv2.putText(vis, label, (int(x1) + 3, int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # 궤적 꼬리: 매 프레임 중심점을 이어 그리기 (예제 방식). 최근 1초 안에 보인 ID만 그린다.
        for tid, pts in history.items():
            if fidx - last_seen.get(tid, -10**9) > fps:
                continue
            tail = np.array(pts[-args.tail:], dtype=np.int32).reshape(-1, 1, 2)
            if len(tail) > 1:
                cv2.polylines(vis, [tail], False, color(tid), 4)
            cv2.circle(vis, tuple(tail[-1][0]), 6, color(tid), -1)
        cv2.putText(vis, f"YOLOv8n (bird/cat/dog) + {Path(args.tracker).stem}  t={(fidx - first) / fps:4.1f}s",
                    (15, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(vis)
    writer.release()
    cap.release()

    # 1) 궤적 CSV 저장 (교수님 메모: trajectory를 따로 저장)
    with open(out / "trajectories.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_sec", "track_id", "class", "conf", "cx", "cy", "x1", "y1", "x2", "y2"])
        w.writerows(rows)

    # 2) 전체 궤적 그림
    canvas = (background * 0.5).astype(np.uint8) if background is not None else np.zeros((H, W, 3), np.uint8)
    for tid, pts in history.items():
        p = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        if len(p) > 1:
            cv2.polylines(canvas, [p], False, color(tid), 5)
        cv2.circle(canvas, tuple(p[0][0]), 9, color(tid), 2)      # 시작점 (빈 원)
        cv2.circle(canvas, tuple(p[-1][0]), 9, color(tid), -1)    # 끝점 (채운 원)
        cv2.putText(canvas, f"{track_cls[tid]} #{tid}", (int(p[-1][0][0]) + 12, int(p[-1][0][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color(tid), 2)
    cv2.imwrite(str(out / "trajectories.png"), canvas)

    # 3) 요약
    summary = {
        "source": str(src), "model": args.model, "tracker": args.tracker, "conf": args.conf, "agnostic_nms": args.agnostic_nms,
        "fps": fps, "start_sec": args.start, "frames": processed,
        "tracks": {
            str(tid): {
                "class": track_cls[tid], "frames_seen": len(pts),
                "path_length_px": round(float(np.sum(np.linalg.norm(np.diff(np.array(pts), axis=0), axis=1))) if len(pts) > 1 else 0.0, 1),
            } for tid, pts in history.items()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
