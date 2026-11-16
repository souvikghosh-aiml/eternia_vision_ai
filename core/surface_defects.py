def process_surface_inspection(model, img):
    """
    Runs YOLO inference on the image and extracts bounding box data.
    """
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
            
    return detections