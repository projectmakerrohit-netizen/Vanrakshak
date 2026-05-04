import os
import tempfile
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
import cv2

from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO

print("Loading models...")

# ==========================
# LOAD MODELS
# ==========================
general_model = YOLO("yolov8n.pt")
gun_model = YOLO("firearm.pt")

interpreter = tf.lite.Interpreter(model_path="yamnet.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = pd.read_csv("yamnet_class_map.csv")
class_names = labels["display_name"].tolist()

EXPECTED_AUDIO_LEN = input_details[0]["shape"][0]

print("Models loaded successfully.")
print("YAMNet expects:", EXPECTED_AUDIO_LEN)

# ==========================
# FASTAPI APP
# ==========================
app = FastAPI(
    title="Vanraksha ML API",
    description="Wildlife / Poacher Detection Service",
    version="1.0.0"
)

# ==========================
# ROOT ROUTE
# ==========================
@app.get("/")
def home():
    return {
        "service": "Vanraksha ML API",
        "status": "running",
        "routes": [
            "/",
            "/health",
            "/docs",
            "/analyze-image",
            "/analyze-audio"
        ]
    }


# ==========================
# HEALTH
# ==========================
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models": [
            "yolov8n",
            "firearm",
            "yamnet"
        ]
    }


# ==========================
# ANALYZE IMAGE
# ==========================
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        npimg = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return {
                "success": False,
                "error": "Invalid image"
            }

        detected_objects = []
        gun_found = False
        best_conf = 0

        # ----------------------
        # YOLO GENERAL
        # ----------------------
        results = general_model(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if conf > 0.4:
                    label = general_model.names[cls]
                    detected_objects.append(label)
                    best_conf = max(best_conf, conf)

        # ----------------------
        # FIREARM
        # ----------------------
        gun_results = gun_model(frame, verbose=False)

        for r in gun_results:
            for box in r.boxes:
                conf = float(box.conf[0])

                if conf > 0.35:
                    gun_found = True
                    best_conf = max(best_conf, conf)

        detected_objects = list(set(detected_objects))

        # ----------------------
        # RISK ENGINE
        # ----------------------
        detected_type = "none"
        risk = "low"

        if "person" in detected_objects and gun_found:
            detected_type = "poacher"
            risk = "high"

        elif "person" in detected_objects:
            detected_type = "human"
            risk = "medium"

        elif gun_found:
            detected_type = "weapon"
            risk = "high"

        elif len(detected_objects) > 0:
            detected_type = "animal"
            risk = "low"

        return {
            "success": True,
            "type": detected_type,
            "objects": detected_objects,
            "gun_detected": gun_found,
            "confidence": round(float(best_conf), 3),
            "risk": risk
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ==========================
# ANALYZE AUDIO
# ==========================
@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    temp_path = None

    try:
        suffix = os.path.splitext(file.filename)[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        audio, sr = librosa.load(
            temp_path,
            sr=16000,
            mono=True
        )

        if len(audio) < EXPECTED_AUDIO_LEN:
            pad = EXPECTED_AUDIO_LEN - len(audio)
            audio = np.pad(audio, (0, pad))
        else:
            audio = audio[:EXPECTED_AUDIO_LEN]

        audio = audio.astype(np.float32)

        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        interpreter.set_tensor(
            input_details[0]["index"],
            audio
        )

        interpreter.invoke()

        scores = interpreter.get_tensor(
            output_details[0]["index"]
        )

        mean_scores = np.mean(scores, axis=0)
        top5_idx = np.argsort(mean_scores)[-5:][::-1]

        top5 = []
        for idx in top5_idx:
            top5.append({
                "class": class_names[idx],
                "score": round(float(mean_scores[idx]), 3)
            })

        top_class = top5_idx[0]
        sound = class_names[top_class]
        confidence = float(mean_scores[top_class])

        risk = "low"

        risky_sounds = [
            "Chainsaw",
            "Gunshot, gunfire",
            "Explosion",
            "Scream",
            "Yell"
        ]

        if sound in risky_sounds:
            risk = "high"

        return {
            "success": True,
            "sound": sound,
            "confidence": round(confidence, 3),
            "top5": top5,
            "risk": risk
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
