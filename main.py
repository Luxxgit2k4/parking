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


logger = logging.getLogger("logs") # idhu vandhu logging ku da
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

path = os.path.dirname(os.path.abspath(__file__)) # dynamic path loader so nee path hardcode panna avasiyam ila
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

def connectDB() -> Any:   #inga dhaan database connect panrom
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
                slot_id VARCHAR(10) PRIMARY KEY,
                slot_status BOOLEAN
            );
        """)
        logger.info("Table 'slots' created or already exists.")
        return conn, cur
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None, None


def generate_slot_mapping(rows, columns): # slot mapping da kanaa
    slot_mapping = {}
    for row in range(rows):
        for col in range(columns):
            slot_id = f"{chr(65 + row)}{str(col + 1).zfill(2)}"
            slot_mapping[(row * columns) + col] = slot_id
    return slot_mapping

try:
    _ = model.names
except AttributeError:
    logger.error("Error: Model does not have 'names' attribute. Check loading process.")
    exit()

def process_detections(frame, results, confidence_threshold=0.2, slot_mapping=None):
    total_spaces = 0   # threshold increase panna accuracy erum but leave it as it is
    filled_spaces = 0
    data = []
    for result in results.xyxy[0]:
        x1, y1, x2, y2, confidence, cls = result.cpu().numpy()
        if confidence < confidence_threshold: # skips detection if threshold is less
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls = int(cls)
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

    update_slots_in_db(data, slot_mapping)

    not_filled_spaces = total_spaces - filled_spaces
    output = {
        "Total spaces": total_spaces,
        "Filled": filled_spaces,
        "Not Filled": not_filled_spaces,
        "Data": data
    }

    return frame, output

def update_slots_in_db(data, slot_mapping):
    conn, cur = connectDB()
    if conn and cur:
        try:
            query = "INSERT INTO slots (slot_id, slot_status) VALUES (%s, %s) ON CONFLICT (slot_id) DO UPDATE SET slot_status = EXCLUDED.slot_status"
            for idx, status in enumerate(data):
                slot_id = slot_mapping.get(idx)
                status = True if status == 1 else False
                cur.execute(query, (slot_id, status))
            conn.commit()
            logger.info(f"Successfully inserted/updated {len(data)} records.")
        except Exception as e:
            logger.error(f"Error inserting/updating data: {e}")
        finally:
            cur.close()
            conn.close()

@app.get("/parking")
async def startParkingDetection(bgTasks: BackgroundTasks) -> Dict:
    def parkingDetection() -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            url = "http://192.168.1.4:4747/video"  # uncomment it if you use droidcam and paste your link here
            cap = cv2.VideoCapture(url) # default cam in device
            cap.set(3, 640)
            cap.set(4, 480)
            logger.info("Started parking slot detection...")
            start = time.time()
            timeout = 3000
            parkData: Dict = {"Data": []}
            max_runtime = 150
            end_time = time.time() + max_runtime
            rows = 2
            columns = 3
            slot_mapping = generate_slot_mapping(rows, columns)  # dynamic slot mapping

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
                frame_with_boxes, parkData = process_detections(frame, results, slot_mapping=slot_mapping)
                logger.info(f"Park Data: {parkData}")
                logger.info(f"Inside park data: {parkData['Data']}")

                # if parkData["Data"]:
                #     conn, cur = connectDB()
                #     if conn and cur:
                #         query = "INSERT INTO slots (slot_id, slot_status) VALUES (%s, %s) ON CONFLICT (slot_id) DO UPDATE SET slot_status = EXCLUDED.slot_status"
                #         try:
                #             for idx, status in enumerate(parkData["Data"]):
                #                 slot_id = slot_mapping.get(idx)
                #                 # Convert 0 to False and 1 to True for database insertion
                #                 status = True if status == 1 else False
                #                 cur.execute(query, (slot_id, status))
                #             conn.commit()
                #             logger.info(f"Successfully inserted {len(parkData['Data'])} records.")
                #         except Exception as e:
                #             logger.error(f"Error inserting/updating data: {e}")
                #         finally:
                #             cur.close()
                #             conn.close()

            cap.release()

    bgTasks.add_task(parkingDetection)
    logger.info("Started background task for parking slot detection.")
    return {"message": "Started parking slot detection..."}

@app.get("/parkingData")
async def getParkingData() -> Dict:
    conn, cur = connectDB()
    if not conn or not cur:
        return {"error": "Failed to connect to the database"}
    try:
        cur.execute("SELECT * FROM slots;")
        records = cur.fetchall()
        cur.close()
        conn.close()
        slots = [{"slot_id": row[0], "slot_status": row[1]} for row in records]
        return {"slots": slots}
    except Exception as e:
        return {"error": str(e)}

def cleardata():
    conn, cur = connectDB()
    if not conn or not cur:
        logger.error("Failed to connect to the database during data clearing.")
        return
    try:
        cur.execute("DELETE FROM slots;")
        conn.commit()
        logger.info("All slot data has been cleared from the database.")
    except Exception as e:
        logger.error(f"Error clearing data: {e}")
    finally:
        cur.close()
        conn.close()

Cleardata = True

if __name__ == "__main__":
    if Cleardata:
        cleardata()
    uvicorn.run(app, host="127.0.0.1", port=8004)
