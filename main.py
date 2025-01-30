import os
import cv2
import torch
import numpy as np
import psycopg2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import Any
from pydantic import BaseModel
from psycopg2 import sql

# Load the YOLOv5 model
model: Any = torch.hub.load('/home/lakshmanan/projects/parkingspace/yolov5', 'custom',
                            path='/home/lakshmanan/projects/parkingspace/model/best.pt',
                            source='local')

if hasattr(model, 'names'):
    print("Model loaded successfully, class names available.")
else:
    print("Error: Model does not have 'names' attribute. Check loading process.")
    exit()

# FastAPI app
app = FastAPI()

# Hardcoded database connection details (without password)
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "parkingspace_db"

def create_database():
    try:
        # Connect to the PostgreSQL server (without password, assuming local setup)
        conn = psycopg2.connect(f"postgresql://localhost:{DB_PORT}")
        conn.autocommit = True
        cur = conn.cursor()

        # Check if database exists, create it if not
        cur.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{DB_NAME}'")
        exists = cur.fetchone()
        if not exists:
            cur.execute(f"CREATE DATABASE {DB_NAME}")
            print(f"Database '{DB_NAME}' created successfully.")
        conn.close()

    except Exception as e:
        print(f"Error creating database: {e}")
        exit()

def create_table():
    try:
        conn = psycopg2.connect(f"postgresql://localhost:{DB_PORT}/{DB_NAME}")
        cur = conn.cursor()

        # Create table if it does not exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS parking_data (
                id SERIAL PRIMARY KEY,
                total_spaces INT,
                filled_spaces INT,
                not_filled_spaces INT,
                data INT[]
            );
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error creating table: {e}")
        exit()

# Create database and table on startup
create_database()
create_table()

def process_detections(frame, results, confidence_threshold=0.3):
    total_spaces = 0
    filled_spaces = 0
    data = []

    for result in results.xyxy[0]:
        x1, y1, x2, y2, confidence, cls = result.cpu().numpy()

        if confidence < confidence_threshold:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls = int(cls)

        if cls == 0:
            total_spaces += 1
            data.append(0)
            color = (0, 0, 255)  # Red for empty spaces
        elif cls == 1:
            total_spaces += 1
            filled_spaces += 1
            data.append(1)
            color = (0, 255, 0)  # Green for occupied spaces
        else:
            continue

        label = f"{model.names[cls]} {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    not_filled_spaces = total_spaces - filled_spaces
    output = {
        "Total spaces": total_spaces,
        "Filled": filled_spaces,
        "Not Filled": not_filled_spaces,
        "Data": data
    }

    # Insert data into the PostgreSQL database
    insert_data(output)

    return frame, output

def insert_data(parking_data):
    try:
        conn = psycopg2.connect(f"postgresql://localhost:{DB_PORT}/{DB_NAME}")
        cur = conn.cursor()

        query = sql.SQL("""
            INSERT INTO parking_data (total_spaces, filled_spaces, not_filled_spaces, data)
            VALUES (%s, %s, %s, %s);
        """)
        cur.execute(query, (
            parking_data["Total spaces"],
            parking_data["Filled"],
            parking_data["Not Filled"],
            parking_data["Data"]
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error inserting data: {e}")
        conn.rollback()

def generate_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera or video feed.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to retrieve frame from feed.")
            break

        # Convert BGR to RGB before passing to model
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Ensure model is callable
        try:
            results = model(rgb_frame)
        except TypeError as e:
            print(f"Error: Model inference failed. {e}")
            break

        frame_with_boxes, parking_data = process_detections(frame, results)
        print(parking_data)

        # Encode the frame in JPEG
        ret, buffer = cv2.imencode('.jpg', frame_with_boxes)
        if not ret:
            continue

        # Convert the frame to bytes and yield it as a response
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/parkingspace")
def parking_space():
    # This triggers the camera feed and stores data
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return {"error": "Unable to access the camera."}

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return {"error": "Unable to retrieve frame from feed."}

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        results = model(rgb_frame)
    except TypeError as e:
        cap.release()
        return {"error": f"Model inference failed. {e}"}

    frame_with_boxes, parking_data = process_detections(frame, results)
    cap.release()

    return parking_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
