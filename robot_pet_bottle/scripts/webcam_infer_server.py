#!/usr/bin/env python3
"""
웹캠 + CNN 상단 바 + YOLOv8n(person/bottle) 박스 — Flask MJPEG 스트림.
접속: http://<라즈베리파이IP>:8765/
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import torch
from flask import Flask, Response
from torchvision import transforms

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from infer_cnn_mixed import load_model  # noqa: E402

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def build_frames(
    cap,
    model,
    classes,
    device,
    tfm,
    yolo_model,
    conf_thres: float,
    lock,
    frame_holder: dict,
):
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        h, w = frame.shape[:2]
        bar_h = max(28, int(h * 0.08))
        pil = transforms.functional.to_pil_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        x = tfm(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[0]
        top_idx = int(prob.argmax().item())
        top_name = classes[top_idx]
        top_p = float(prob[top_idx].item())

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        label = f"{top_name}  {100 * top_p:.1f}%"
        cv2.putText(
            frame,
            label,
            (12, int(bar_h * 0.72)),
            cv2.FONT_HERSHEY_SIMPLEX,
            min(w, h) / 900.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if yolo_model is not None:
            try:
                res = yolo_model(frame, conf=conf_thres, verbose=False)[0]
                names = res.names
                for b in res.boxes:
                    xyxy = b.xyxy[0].cpu().numpy()
                    cid = int(b.cls[0].item()) if b.cls is not None else -1
                    cname = names.get(cid, str(cid)) if isinstance(names, dict) else str(cid)
                    if cname not in ("person", "bottle"):
                        continue
                    x1, y1, x2, y2 = map(int, xyxy)
                    color = (0, 255, 128) if cname == "person" else (255, 180, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        cname,
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
            except Exception:
                pass

        with lock:
            frame_holder["jpg"] = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])[1].tobytes()


def mjpeg_gen(frame_holder, lock):
    while True:
        with lock:
            jpg = frame_holder.get("jpg")
        if jpg is None:
            time.sleep(0.02)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(0.03)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default="./cnn_mixed_work/cnn_mixed_mobilenet_v2.pt")
    ap.add_argument("--yolo", default="yolov8n.pt")
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--yolo-conf", type=float, default=0.35)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = ap.parse_args()

    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = Path(args.model)
    if not ckpt.is_file():
        raise SystemExit(f"CNN 모델 없음: {ckpt}")

    model, classes = load_model(ckpt, device)
    tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    yolo_model = None
    if not args.no_yolo and YOLO is not None:
        yolo_model = YOLO(args.yolo)
    elif not args.no_yolo and YOLO is None:
        print("[WARN] ultralytics 미설치 — YOLO 박스 비활성")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"카메라 열기 실패: index={args.camera}")

    lock = threading.Lock()
    frame_holder = {"jpg": None}
    t = threading.Thread(
        target=build_frames,
        args=(cap, model, classes, device, tfm, yolo_model, args.yolo_conf, lock, frame_holder),
        daemon=True,
    )
    t.start()

    app = Flask(__name__)

    @app.route("/")
    def index():
        return (
            "<html><head><meta charset=utf-8><title>robot_pet_bottle</title></head>"
            "<body style=margin:0;background:#111;color:#eee;font-family:sans-serif>"
            "<div style=padding:8px>CNN + YOLO — <code>/stream</code> MJPEG</div>"
            '<img src="/stream" style=max-width:100%;height:auto />'
            "</body></html>"
        )

    @app.route("/stream")
    @app.route("/video_feed")
    def stream():
        return Response(mjpeg_gen(frame_holder, lock), mimetype="multipart/x-mixed-replace; boundary=frame")

    print(f"[INFO] http://{args.host}:{args.port}/  device={device}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
