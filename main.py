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
from typing import Any, Dict

load_dotenv()
app = FastAPI()

# Configure logger
logger = logging.getLogger("logs")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# Define paths for YOLOv5 and model
path = os.path.dirname(os.path.abspath(__file__))
yolov5path = os.path.join(path, 'yolov5')
modelpath = os.path.join(path, 'model', 'best.pt')

logger.info(f"Found Yolov5 at: {yolov5path}")
logger.info(f"Found the model at: {modelpath}")

# Load YOLOv5 model
logger.info("Loading the model...")
model: Any = torch.hub.load(
    yolov5path,
    'custom',
    path=modelpath,
    source='local'
)

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

def cleardata():
    conn = connectDB()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM slots;")  # Deletes all data in the slots table
            conn.commit()
            cur.close()
            logger.info("Data cleared from 'slots' table.")
        except Exception as e:
            logger.error(f"Error while clearing data: {e}")
        finally:
            conn.close()

# Check if the model is loaded correctly
try:
    _ = model.names
except AttributeError:
    logger.error("Error: Model does not have 'names' attribute. Check loading process.")
    exit()

# Detection logic
def process_detections(frame, results, confidence_threshold=0.3):
    total_spaces = 0
    filled_spaces = 0
    data = []

    # Iterate over the detection results
    for result in results.xyxy[0]:
        x1, y1, x2, y2, confidence, cls = result.cpu().numpy()

        # Skip detections below the confidence threshold
        if confidence < confidence_threshold:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls = int(cls)

        # Determine slot status (empty or filled)
        if cls == 0:
            total_spaces += 1
            data.append(0)  # Empty space
            color = (0, 0, 255)  # Red for empty spaces
        elif cls == 1:
            total_spaces += 1
            filled_spaces += 1
            data.append(1)  # Filled space
            color = (0, 255, 0)  # Green for occupied spaces
        else:
            continue

        # Label and draw the bounding box
        label = f"{model.names[cls]} {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Calculate and return output data
    not_filled_spaces = total_spaces - filled_spaces
    output = {
        "Total spaces": total_spaces,
        "Filled": filled_spaces,
        "Not Filled": not_filled_spaces,
        "Data": data
    }

    return frame, output

# API Endpoints
@app.get("/parking")
async def startParkingDetection(bgTasks: BackgroundTasks) -> Dict:
    def parkingDetection() -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            cap = cv2.VideoCapture(0)
            cap.set(3, 640)  # Set width
            cap.set(4, 480)  # Set height
            logger.info("Started parking slot detection...")
            start = time.time()
            timeout = 30
            parkData: Dict = {"Data": []}

            # Set maximum runtime duration (timeout in seconds)
            max_runtime = 30  # seconds
            end_time = time.time() + max_runtime

            while time.time() < end_time:
                success, frame = cap.read()
                if not success:
                    logger.error("Failed to grab frame")
                    break
                rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                try:
                    results = model(rgbFrame)
                except TypeError as e:
                    logger.error(f"Error during inference: {e}")
                    break

                # Call the detection logic
                frame_with_boxes, parkData = process_detections(frame, results)

                # Log the parkData to check its content
                logger.info(f"Park Data: {parkData}")

                # Insert data continuously
                if parkData["Data"]:
                    logger.info(f"Parking Data: {parkData}")
                    conn = connectDB()
                    if conn:
                        cur = conn.cursor()
                        query = "INSERT INTO slots (slot_status) VALUES (%s)"
                        try:
                            for status in parkData["Data"]:
                                # Convert 0 to False and 1 to True for database insertion
                                status = True if status == 1 else False
                                cur.execute(query, (status,))
                            conn.commit()  # Ensure commit is triggered after all insertions
                            logger.info(f"Successfully inserted {len(parkData['Data'])} records.")
                        except Exception as e:
                            logger.error(f"Error inserting data: {e}")
                        finally:
                            cur.close()
                            conn.close()

            cap.release()

    bgTasks.add_task(parkingDetection)
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

Cleardata = True

if __name__ == "__main__":
    if Cleardata:
        cleardata()
    uvicorn.run(app, host="127.0.0.1", port=8004)
