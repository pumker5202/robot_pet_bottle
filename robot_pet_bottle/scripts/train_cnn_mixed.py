#!/usr/bin/env python3
"""AI Hub + Roboflow 혼합 CNN 학습. ResNet18 또는 MobileNetV2(경량, Pi 배포용)."""
import argparse
import json
import os
import random
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from roboflow import Roboflow
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def list_zip_files(base: Path, prefix: str):
    return sorted([p for p in base.rglob("*.zip") if p.name.startswith(prefix)])


def safe_name(text: str):
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)


def extract_keyframe_from_video(zip_path: Path, inner_video: str, frame_idx: int, out_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(zf.read(inner_video))
            tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        os.unlink(tmp_path)
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    ok, frame = cap.read()
    cap.release()
    os.unlink(tmp_path)
    if not ok or frame is None:
        return False
    cv2.imwrite(str(out_path), frame)
    return True


def prepare_aihub(aihub_base: Path, out_train: Path, ratio: float, seed: int, max_count: int = 0):
    label_zips = list_zip_files(aihub_base / "02_라벨링데이터", "TL_")
    src_zips = list_zip_files(aihub_base / "01_원천데이터", "TS_")
    if not label_zips or not src_zips:
        raise RuntimeError("AI_HUB 라벨 또는 원천 zip을 찾을 수 없습니다.")

    video_index = {}
    for sp in src_zips:
        with zipfile.ZipFile(sp) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".mp4"):
                    video_index[Path(name).name] = (sp, name)

    json_items = []
    for lp in label_zips:
        with zipfile.ZipFile(lp) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".json"):
                    try:
                        obj = json.loads(zf.read(name).decode("utf-8", "ignore"))
                    except Exception:
                        continue
                    fn = obj.get("meta_data", {}).get("file_name")
                    kf = obj.get("annotations", {}).get("keyframe", 0)
                    if fn and fn in video_index:
                        json_items.append((fn, int(kf), video_index[fn]))

    if not json_items:
        raise RuntimeError("라벨-원천 매칭 가능한 항목이 없습니다.")

    random.Random(seed).shuffle(json_items)
    n = max(1, int(len(json_items) * ratio))
    if max_count > 0:
        n = min(n, max_count)
    picked = json_items[:n]

    cls_dir = out_train / "aihub_action"
    cls_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for i, (video_name, keyframe, (zip_path, inner_video)) in enumerate(picked):
        out_img = cls_dir / f"aihub_{i:06d}_{safe_name(video_name)}.jpg"
        ok = extract_keyframe_from_video(zip_path, inner_video, keyframe, out_img)
        if ok:
            made += 1
    print(f"[AIHUB] matched={len(json_items)} sampled={n} extracted={made}")


def yolo_txt_to_xyxy(label_line, w, h):
    parts = label_line.strip().split()
    if len(parts) != 5:
        return None
    cls_id = int(float(parts[0]))
    cx, cy, bw, bh = map(float, parts[1:])
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return cls_id, x1, y1, x2, y2


def prepare_roboflow(out_train: Path, api_key: str, workspace: str, project: str, version: int):
    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(workspace).project(project).version(version).download("yolov8")
    root = Path(ds.location)
    yaml_path = root / "data.yaml"
    names = {}
    if yaml_path.exists():
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
            names_raw = y.get("names", {})
            if isinstance(names_raw, list):
                names = {i: n for i, n in enumerate(names_raw)}
            elif isinstance(names_raw, dict):
                names = {int(k): v for k, v in names_raw.items()}

    made = 0
    for split in ("train", "valid", "test"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        for img_path in img_dir.glob("*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            txt = lbl_dir / (img_path.stem + ".txt")
            if not txt.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            lines = txt.read_text(encoding="utf-8", errors="ignore").splitlines()
            for idx, line in enumerate(lines):
                item = yolo_txt_to_xyxy(line, w, h)
                if item is None:
                    continue
                cls_id, x1, y1, x2, y2 = item
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                cls_name = safe_name(str(names.get(cls_id, f"class_{cls_id}")))
                cls_dir = out_train / cls_name
                cls_dir.mkdir(parents=True, exist_ok=True)
                out = cls_dir / f"rf_{split}_{img_path.stem}_{idx}.jpg"
                cv2.imwrite(str(out), crop)
                made += 1
    print(f"[ROBOFLOW] cropped_samples={made} from {root}")


def load_latest_checkpoint(ckpt_dir: Path):
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not ckpts:
        return None
    return ckpts[-1]


def build_model(backbone: str, num_classes: int):
    if backbone == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    elif backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"unknown backbone: {backbone}")
    return m


def train_cnn(
    train_dir: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    backbone: str,
    resume: bool = False,
    device_override: str = "auto",
):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    ds = datasets.ImageFolder(str(train_dir), transform=transform)
    if len(ds.classes) < 2:
        raise RuntimeError("클래스가 2개 이상이어야 CNN 분류 학습이 가능합니다.")

    if device_override == "cpu":
        device = "cpu"
    elif device_override == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device} backbone={backbone}")

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    opt = None
    model = None

    if resume:
        latest = load_latest_checkpoint(ckpt_dir)
        if latest is not None:
            state = _torch_load(latest, device)
            backbone = state.get("backbone", backbone)
            model = build_model(backbone, len(ds.classes))
            model.load_state_dict(state["model_state_dict"])
            model = model.to(device)
            opt = optim.Adam(model.parameters(), lr=lr)
            opt.load_state_dict(state["optimizer_state_dict"])
            start_epoch = int(state["epoch"]) + 1
            print(f"[RESUME] {latest.name} next_epoch={start_epoch} backbone={backbone}")
        else:
            print("[RESUME] checkpoint 없음")
            model = build_model(backbone, len(ds.classes)).to(device)
            opt = optim.Adam(model.parameters(), lr=lr)
    else:
        model = build_model(backbone, len(ds.classes)).to(device)
        opt = optim.Adam(model.parameters(), lr=lr)

    crit = nn.CrossEntropyLoss()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model.train()
    for ep in range(start_epoch, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        total_steps = len(loader)
        for step, (x, y) in enumerate(loader, start=1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * x.size(0)
            pred = out.argmax(1)
            correct += int((pred == y).sum().item())
            total += int(y.size(0))
            if step == 1 or step % 20 == 0 or step == total_steps:
                pct = 100.0 * step / max(1, total_steps)
                print(f"[GAUGE] epoch={ep}/{epochs} step={step}/{total_steps} ({pct:.1f}%)", flush=True)
        print(f"[TRAIN] epoch={ep}/{epochs} loss={total_loss/max(1,total):.4f} acc={correct/max(1,total):.4f}")
        torch.save(
            {
                "epoch": ep,
                "backbone": backbone,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "classes": ds.classes,
            },
            ckpt_dir / f"epoch_{ep:03d}.pt",
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"cnn_mixed_{backbone}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": ds.classes,
            "backbone": backbone,
        },
        out_dir / out_name,
    )
    (out_dir / "classes.txt").write_text("\n".join(ds.classes), encoding="utf-8")
    print(f"[DONE] model saved to {out_dir / out_name}")


def _torch_load(path: Path, map_loc):
    try:
        return torch.load(path, map_location=map_loc, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_loc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aihub-base", default="./data", help="01_원천데이터, 02_라벨링데이터 상위 폴더")
    p.add_argument("--work-dir", default="./cnn_mixed_work")
    p.add_argument("--rf-api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""))
    p.add_argument("--rf-workspace", default="senior-project-g6ziw")
    p.add_argument("--rf-project", default="water-bottle-n1qpn")
    p.add_argument("--rf-version", type=int, default=3)
    p.add_argument("--aihub-ratio", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--aihub-max-count", type=int, default=2000)
    p.add_argument(
        "--backbone",
        choices=("resnet18", "mobilenet_v2"),
        default="mobilenet_v2",
        help="mobilenet_v2: 경량(라즈베리파이 배포용), resnet18: 더 무겁고 보통 더 정확",
    )
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-prepare", action="store_true")
    args = p.parse_args()

    work = Path(args.work_dir)
    train_dir = work / "train"
    if not args.resume and not args.skip_prepare:
        if work.exists():
            shutil.rmtree(work)
        train_dir.mkdir(parents=True, exist_ok=True)
        prepare_aihub(Path(args.aihub_base), train_dir, args.aihub_ratio, args.seed, args.aihub_max_count)
        if not args.rf_api_key:
            raise RuntimeError("ROBOFLOW_API_KEY 를 환경변수로 설정해 주세요.")
        prepare_roboflow(train_dir, args.rf_api_key, args.rf_workspace, args.rf_project, args.rf_version)
    else:
        if not train_dir.exists():
            raise RuntimeError("resume/skip-prepare 모드인데 train 데이터가 없습니다.")
    train_cnn(
        train_dir,
        work,
        args.epochs,
        args.batch_size,
        args.lr,
        args.backbone,
        resume=args.resume,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
