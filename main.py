import logging
import warnings
import sys
import threading
import cv2
import torch
import numpy as np
import psycopg2
import uvicorn
from fastapi import BackgroundTasks, FastAPI
from dotenv import load_dotenv
import os
import time
from typing import Any, Tuple, Dict

load_dotenv()
app = FastAPI()

logger = logging.getLogger("logs")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

path = os.path.dirname(os.path.abspath(__file__))

yolov5path = os.path.join(path, 'yolov5')
modelpath = os.path.join(path, 'model', 'best.pt')

logger.info(f"Found Yolov5 at: {yolov5path}")
logger.info(f"Found the model at: {modelpath}")

logger.info("Loading the model...")
model: Any = torch.hub.load(
    yolov5path,
    'custom',
    path=modelpath,
    source='local'
)

try:
    _ = model.names
except AttributeError:
    logger.error("Error: Model does not have 'names' attribute. Check loading process.")
    exit()

def connectDB() -> Any:
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS slots (
                slot_id SERIAL PRIMARY KEY,
                slot_status BOOLEAN,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        logger.info("Table 'slots' created or already exists.")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def cropROI(frame: np.ndarray) -> Tuple[np.ndarray, int]:
    height, width, _ = frame.shape
    roi = frame[int(height * 0.5):height, 0:width]
    return roi, int(height * 0.5)

def processDetections(frame: np.ndarray, results: Any, threshold: float = 0.5) -> Tuple[np.ndarray, Dict]:
    total = 0
    filled = 0
    data = []
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result.cpu().numpy()
        if conf < threshold:
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls = int(cls)
        w, h = x2 - x1, y2 - y1
        if w < 50 or h < 50:
            continue
        ratio = w / h if h != 0 else 0
        if not (1.5 < ratio < 5):
            continue
        if cls == 0:
            total += 1
            data.append(0)
            color = (0, 0, 255)
        elif cls == 1:
            total += 1
            filled += 1
            data.append(1)
            color = (0, 255, 0)
        else:
            continue
        label = f"{model.names[cls]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    not_filled = total - filled
    output = {"Total spaces": total, "Filled": filled, "Not Filled": not_filled, "Data": data}
    return frame, output

def parkingDetection(bgTasks: BackgroundTasks) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        logger.info("Started parking slot detection...")
        start = time.time()
        timeout = 30
        parkData: Dict = {"Data": []}
        while True:
            success, frame = cap.read()
            if not success:
                logger.error("Failed to grab frame")
                break
            roiFrame, yOff = cropROI(frame)
            rgbFrame = cv2.cvtColor(roiFrame, cv2.COLOR_BGR2RGB)
            try:
                results = model(rgbFrame)
            except TypeError as e:
                logger.error(f"Error during inference: {e}")
                break
            processed, parkData = processDetections(roiFrame, results)
            if parkData["Data"]:
                logger.info(f"Parking Data: {parkData}")
            if time.time() - start > timeout:
                logger.info("Stopping parking slot detection due to timeout...")
                break
        cap.release()
        if parkData["Data"]:
            conn = connectDB()
            if conn:
                cur = conn.cursor()
                query = "INSERT INTO slots (slot_status) VALUES (%s)"
                for status in parkData["Data"]:
                    cur.execute(query, (status,))
                conn.commit()
                cur.close()
                conn.close()
                logger.info("Parking data inserted into database.")
@app.get("/parking")
async def startParkingDetection(bgTasks: BackgroundTasks) -> Dict:
    bgTasks.add_task(parkingDetection, bgTasks)
    logger.info("Started background task for parking slot detection.")
    return {"message": "Started parking slot detection..."}

@app.get("/parkingData")
async def getParkingData() -> Dict:
    conn = connectDB()
    if not conn:
        return {"error": "Failed to connect to the database"}
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM slots;")
        records = cur.fetchall()
        cur.close()
        conn.close()
        slots = [{"slot_id": row[0], "slot_status": row[1], "timestamp": row[2]} for row in records]
        return {"slots": slots}
    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8004)
