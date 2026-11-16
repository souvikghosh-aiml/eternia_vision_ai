from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
import cv2
import numpy as np
import io

app = FastAPI(title="Eternia PDI - Surface Defect API", version="1.0")

model = None

@app.on_event("startup")
def load_model():
    """Loads the YOLO model into memory when the server starts."""
    global model
    try:
        model = YOLO("eternia_training/Eternia_Training/runs/surface_defects_prod_v1-2/weights/best.pt")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/api/v1/inspect/surface")
async def inspect_surface(file: UploadFile = File(...)):
    """
    Accepts an image file, runs YOLO inference, and returns bounding boxes.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format.")

        results = model.predict(img, conf=0.25)

        detections = []
        for r in results:
            for box in r.boxes:
                bbox = box.xyxy.tolist()[0]
                detections.append({
                    "class_name": model.names[int(box.cls)],
                    "class_id": int(box.cls),
                    "confidence": float(box.conf),
                    "bbox": {
                        "xmin": round(bbox[0], 2),
                        "ymin": round(bbox[1], 2),
                        "xmax": round(bbox[2], 2),
                        "ymax": round(bbox[3], 2)
                    }
                })

        return {
            "status": "success",
            "image_filename": file.filename,
            "total_defects_found": len(detections),
            "detections": detections
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))