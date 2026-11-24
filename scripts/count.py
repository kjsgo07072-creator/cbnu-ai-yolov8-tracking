"""학습한 YOLOv8 모델 + Ultralytics ObjectCounter로 선을 지나가는 객체 수 세기.

Ultralytics object_counting 예제(solutions.ObjectCounter)를 우리 모델과 영상에 맞게 적용했다.
선을 지나간 객체를 추적 ID 기준으로 한 번씩만 센다 (IN/OUT 방향 구분).

python scripts/count.py --source videos/input_dogs_walk_10s.mp4 --line-y 0.55

결과 (outputs/<name>_count/):
    counting.mp4   박스 + ID + 카운팅 선 + IN/OUT 수
    counts.json    전체 및 클래스별 IN/OUT 수
"""
import argparse
import json
from pathlib import Path

import cv2
from ultralytics import solutions

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True)
ap.add_argument("--model", default=str(ROOT / "01_trained_model/best.pt"))
ap.add_argument("--tracker", default="bytetrack.yaml")
ap.add_argument("--conf", type=float, default=0.1)
ap.add_argument("--line-y", type=float, default=0.55, help="가로 카운팅 선의 높이 (화면 높이 대비 비율)")
ap.add_argument("--name", default=None)
args = ap.parse_args()

src = Path(args.source)
out = ROOT / "outputs" / f"{args.name or src.stem}_count"
out.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(src))
fps, W, H = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
y = int(H * args.line_y)
region = [(0, y), (W, y)]  # 화면을 가로지르는 선

counter = solutions.ObjectCounter(
    model=args.model, region=region, tracker=args.tracker, conf=args.conf,
    show=False, show_in=True, show_out=True, line_width=3, verbose=False,
)
writer = cv2.VideoWriter(str(out / "counting_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
while True:
    ok, frame = cap.read()
    if not ok:
        break
    res = counter(frame)
    writer.write(res.plot_im)
writer.release()
cap.release()

result = {
    "source": str(src), "model": args.model, "tracker": args.tracker, "conf": args.conf,
    "line": region, "in_count": counter.in_count, "out_count": counter.out_count,
    "classwise": {k: dict(v) for k, v in counter.classwise_count.items()},
}
(out / "counts.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(json.dumps(result, indent=2, ensure_ascii=False))
