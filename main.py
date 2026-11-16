from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from contextlib import asynccontextmanager
from ultralytics import YOLO
import cv2
import numpy as np
import json

from core.calibration import calibrate, refine_outline, rectangular_outline, measure
from core.surface_defects import process_surface_inspection

defect_model = None
seg_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads all AI models into memory when the server starts."""
    global defect_model, seg_model
    try:
        defect_model = YOLO("best.pt")
        seg_model = YOLO("yolov8n-seg.pt")
        print("All models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
    yield
    print("Shutting down models.")

app = FastAPI(title="Eternia PDI API", version="1.1", lifespan=lifespan)

@app.post("/api/v1/inspect/surface")
async def inspect_surface(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format.")

        detections = process_surface_inspection(defect_model, img)

        return {
            "status": "success",
            "image_filename": file.filename,
            "total_defects_found": len(detections),
            "detections": detections
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/inspect/measure")
async def measure_image(
    image: UploadFile = File(...), 
    calibration: str = Form(...)
):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format.")
        
    try:
        data = json.loads(calibration)
        points = np.asarray(data["points"], dtype=float)
        ticks = np.asarray(data["ticks"], dtype=float)
        h, w = img.shape[:2]
        
        if points.shape != (4, 2) or ticks.shape != (4,):
            raise HTTPException(status_code=400, detail="Invalid calibration data array shape.")
        if not np.isfinite(points).all() or not np.isfinite(ticks).all():
            raise HTTPException(status_code=400, detail="Calibration must contain finite numbers.")
        if (points < 0).any() or (points >= [w, h]).any():
            raise HTTPException(status_code=400, detail="Calibration points outside image bounds.")
        if "size" in data and data["size"] != [h, w]:
            raise HTTPException(status_code=400, detail=f"Calibration image size mismatch. Expected {data['size']}, got [{h}, {w}]")
            
        perspective = data.get("perspective", False)
        if not isinstance(perspective, bool):
            raise HTTPException(status_code=400, detail="Invalid perspective boolean value.")

        matrix = calibrate(points, ticks, perspective)
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Calibration data must be valid JSON string.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Calibration Error: {str(e)}")

    prediction = seg_model.predict(img, conf=0.4, verbose=False)[0]
    outlines = []
    
    if prediction.masks is not None:
        for outline in prediction.masks.xy:
            if len(outline) >= 3:
                outlines.append(refine_outline(img, outline))
                
    if not outlines:
        rectangle = rectangular_outline(img)
        if rectangle is not None:
            outlines.append(rectangle)

    objects = []
    for outline in outlines:
        try:
            # Delegate to calibration module
            length, width = measure(outline, matrix)
            objects.append({
                "length_mm": round(length, 1),
                "width_mm": round(width, 1)
            })
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f"Measurement Error: {str(e)}")

    if not objects:
        raise HTTPException(status_code=422, detail="No objects detected in the image.")

    return {
        "status": "success",
        "measurements": objects[0] if len(objects) == 1 else objects
    }