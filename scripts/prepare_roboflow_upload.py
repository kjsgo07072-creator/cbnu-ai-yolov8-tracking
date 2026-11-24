"""COCO 2017 train에서 dog / cat / bird 이미지를 부류당 N장 골라 Roboflow 업로드용 폴더를 만든다.

이미지만 넣고 라벨은 넣지 않는다 (라벨링은 Roboflow에서 직접 한다).
COCO 원본 라벨은 나중에 직접 그린 라벨과 비교할 수 있게 따로 저장한다.

사용법:
    python scripts/prepare_roboflow_upload.py --per-class 100

결과:
    roboflow_upload/images/*.jpg        <- Roboflow에 이 폴더를 올린다
    roboflow_upload/coco_reference/*.txt (YOLO 포맷, 업로드하지 않음)
    roboflow_upload/selection.csv        (파일명, 어떤 부류로 뽑혔는지)
"""
import argparse
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from collect_coco import CLASSES, ROOT, download, load, pick, to_yolo

OUT = ROOT / "roboflow_upload"
CACHE = ROOT / "datasets" / "pets" / "images" / "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    imgs, anns = load("train")
    chosen = pick(imgs, anns, args.per_class, random.Random(args.seed))

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "coco_reference").mkdir(parents=True, exist_ok=True)
    rows, jobs = ["file,selected_class"], []
    for img_id, cls in chosen.items():
        im = imgs[img_id]
        dst = OUT / "images" / im["file_name"]
        cached = CACHE / im["file_name"]
        if cached.exists():
            shutil.copy2(cached, dst)
        else:
            jobs.append((im["coco_url"], dst))
        (OUT / "coco_reference" / f"{Path(im['file_name']).stem}.txt").write_text(to_yolo(im, anns[img_id]))
        rows.append(f"{im['file_name']},{CLASSES[cls]}")

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda j: download(*j), jobs))
    (OUT / "selection.csv").write_text("\n".join(rows) + "\n")
    print(f"{len(chosen)} images ({len(jobs)} newly downloaded) -> {OUT / 'images'}")


if __name__ == "__main__":
    main()
