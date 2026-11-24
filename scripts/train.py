"""Roboflow에서 내보낸 데이터셋(YOLOv8 포맷)으로 YOLOv8n 전이학습.

데이터셋: Roboflow Universe에서 YOLOv8 포맷 zip을 받아 datasets/roboflow_v1/ 에 압축을 푼다.
    https://universe.roboflow.com/s-workspace-5ejf0/pets-dog-cat-bird/dataset/1

python scripts/train.py --epochs 100
(최종 모델 설정: epochs=100, patience=100, imgsz=640, batch=16, device=mps)
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="yolov8n.pt", help="COCO 사전학습 가중치")
ap.add_argument("--data-dir", default=str(ROOT / "datasets" / "roboflow_v1"))
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--patience", type=int, default=100, help="조기 종료 기준 (첫 시도는 20이라 21 에폭에서 멈춤)")
ap.add_argument("--device", default="mps", help="Apple GPU=mps, NVIDIA=0, CPU=cpu")
ap.add_argument("--freeze", type=int, default=0, help="앞쪽 N개 레이어 고정 (YOLOv8 백본=10)")
ap.add_argument("--name", default="pets_yolov8n")
args = ap.parse_args()

# Roboflow data.yaml의 상대경로(../train/images)를 쓰지 않고, 절대경로로 학습용 yaml을 만든다.
data_dir = Path(args.data_dir).resolve()
data_yaml = ROOT / "datasets" / "pets_roboflow.yaml"
data_yaml.parent.mkdir(parents=True, exist_ok=True)
data_yaml.write_text(
    f"path: {data_dir}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
    "nc: 3\nnames: ['bird', 'cat', 'dog']\n"
)

model = YOLO(args.model)
model.train(
    data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
    device=args.device, workers=4, seed=0, patience=args.patience, freeze=args.freeze or None,
    project=str(ROOT / "runs"), name=args.name, exist_ok=True,
)

# test 세트 성능
best = YOLO(str(ROOT / "runs" / args.name / "weights" / "best.pt"))
best.val(data=str(data_yaml), split="test", project=str(ROOT / "runs"), name=args.name + "_test", exist_ok=True)
