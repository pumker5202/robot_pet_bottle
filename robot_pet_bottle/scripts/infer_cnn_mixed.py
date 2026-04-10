#!/usr/bin/env python3
"""학습된 CNN(.pt) 추론 — 이미지 경로 / URL / 샘플."""
import argparse
import io
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import torch
from PIL import Image
from torchvision import transforms

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from train_cnn_mixed import build_model  # noqa: E402


def _torch_load(path: Path, map_loc):
    try:
        return torch.load(path, map_location=map_loc, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_loc)


def load_model(ckpt_path: Path, device: str):
    ckpt = _torch_load(ckpt_path, device)
    classes = ckpt.get("classes")
    if not classes:
        raise RuntimeError("체크포인트에 classes 가 없습니다.")
    backbone = ckpt.get("backbone", "resnet18")
    state = ckpt.get("model_state_dict", ckpt)
    model = build_model(backbone, len(classes))
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, classes


def load_image_from_url(url: str, timeout: int = 30):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; infer_cnn_mixed/1.0)"})
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def main():
    p = argparse.ArgumentParser(description="CNN mixed 모델 추론")
    p.add_argument("--model", default="./cnn_mixed_work/cnn_mixed_mobilenet_v2.pt")
    p.add_argument("--image", action="append", default=[])
    p.add_argument("--url", action="append", default=[])
    p.add_argument("--sample", action="store_true")
    p.add_argument("--train-dir", default="./cnn_mixed_work/train")
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args()

    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = Path(args.model)
    if not ckpt_path.is_file():
        raise SystemExit(f"모델 파일 없음: {ckpt_path}")

    model, classes = load_model(ckpt_path, device)
    tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    paths = list(args.image)
    urls = list(args.url)
    if args.sample:
        train_root = Path(args.train_dir)
        for cls_name in classes:
            d = train_root / cls_name
            if not d.is_dir():
                continue
            imgs = sorted(d.glob("*.jpg")) + sorted(d.glob("*.jpeg")) + sorted(d.glob("*.png"))
            if imgs:
                paths.append(str(imgs[0]))

    if not paths and not urls:
        raise SystemExit("--image / --url / --sample 중 하나를 사용하세요.")

    print(f"device={device}")
    print(f"classes ({len(classes)}): {classes}")
    print("-" * 60)

    def run_one(img: Image.Image, label: str):
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[0]
        topk = min(args.topk, len(classes))
        vals, idx = prob.topk(topk)
        print(f"input: {label}")
        for rank in range(topk):
            c = classes[idx[rank].item()]
            pr = vals[rank].item()
            print(f"  #{rank + 1} {c}: {pr:.4f} ({100 * pr:.2f}%)")
        print("-" * 60)

    for img_path in paths:
        path = Path(img_path)
        if not path.is_file():
            print(f"[SKIP] 없음: {path}")
            continue
        img = Image.open(path).convert("RGB")
        run_one(img, str(path))

    for url in urls:
        try:
            img = load_image_from_url(url)
            run_one(img, url)
        except Exception as e:
            print(f"[FAIL] url={url}\n  {type(e).__name__}: {e}")
            print("-" * 60)


if __name__ == "__main__":
    main()
