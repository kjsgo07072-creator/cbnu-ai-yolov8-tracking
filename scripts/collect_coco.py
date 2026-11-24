"""COCO 2017에서 dog / cat / bird 이미지를 골라 내려받고 YOLO 포맷 라벨로 변환한다.

사용법:
    python scripts/collect_coco.py --train-per-class 150 --val-per-class 30

결과:
    datasets/pets/images/{train,val}/*.jpg
    datasets/pets/labels/{train,val}/*.txt   (class cx cy w h, 0~1 정규화)
    datasets/pets/data.yaml
"""
import argparse
import json
import random
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANN_DIR = ROOT / "raw" / "annotations"
OUT = ROOT / "datasets" / "pets"

# COCO category id -> 우리 클래스 id
CLASSES = ["dog", "cat", "bird"]
COCO_IDS = {18: 0, 17: 1, 16: 2}  # dog=18, cat=17, bird=16
MIN_AREA_RATIO = 0.02  # 대표 객체가 이미지의 2% 이상은 되어야 선택


def load(split):
    data = json.loads((ANN_DIR / f"instances_{split}2017.json").read_text())
    imgs = {im["id"]: im for im in data["images"]}
    anns = defaultdict(list)
    for a in data["annotations"]:
        if a["category_id"] in COCO_IDS:
            anns[a["image_id"]].append(a)
    return imgs, anns


def pick(imgs, anns, per_class, rng):
    """클래스마다 per_class장씩, 겹치지 않게 고른다."""
    chosen = {}
    for coco_id, cls in COCO_IDS.items():
        cands = []
        for img_id, alist in anns.items():
            if img_id in chosen:
                continue
            if any(a["iscrowd"] for a in alist):  # crowd 박스가 있으면 라벨이 불완전해서 제외
                continue
            im = imgs[img_id]
            big = [a for a in alist if a["category_id"] == coco_id
                   and a["area"] >= MIN_AREA_RATIO * im["width"] * im["height"]]
            if big:
                cands.append(img_id)
        cands.sort()
        rng.shuffle(cands)
        for img_id in cands[:per_class]:
            chosen[img_id] = cls
    return chosen


def to_yolo(im, alist):
    W, H = im["width"], im["height"]
    lines = []
    for a in alist:
        x, y, w, h = a["bbox"]
        if w < 2 or h < 2:
            continue
        lines.append(f"{COCO_IDS[a['category_id']]} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
    return "\n".join(lines) + "\n"


def download(url, dst):
    if dst.exists() and dst.stat().st_size > 0:
        return
    for _ in range(3):
        try:
            urllib.request.urlretrieve(url, dst)
            return
        except Exception:
            pass
    raise RuntimeError(f"failed: {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-class", type=int, default=150)
    ap.add_argument("--val-per-class", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    summary = {}
    for split, coco_split, n in [("train", "train", args.train_per_class), ("val", "val", args.val_per_class)]:
        imgs, anns = load(coco_split)
        chosen = pick(imgs, anns, n, rng)
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

        jobs = []
        box_count = defaultdict(int)
        for img_id in chosen:
            im = imgs[img_id]
            stem = Path(im["file_name"]).stem
            (OUT / "labels" / split / f"{stem}.txt").write_text(to_yolo(im, anns[img_id]))
            for a in anns[img_id]:
                box_count[CLASSES[COCO_IDS[a["category_id"]]]] += 1
            jobs.append((im["coco_url"], OUT / "images" / split / im["file_name"]))

        with ThreadPoolExecutor(16) as ex:
            list(ex.map(lambda j: download(*j), jobs))

        img_count = defaultdict(int)
        for cls in chosen.values():
            img_count[CLASSES[cls]] += 1
        summary[split] = {"images": len(chosen), "images_by_selected_class": dict(img_count), "boxes": dict(box_count)}
        print(split, summary[split])

    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
