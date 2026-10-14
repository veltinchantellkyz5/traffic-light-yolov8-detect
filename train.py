import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

model = YOLO('yolov8s.pt') 

results = model.train(data='data.yaml', epochs=30, imgsz=320, batch=16, workers=8, device='cpu')
