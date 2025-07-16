
print("Запуск приложения Flask...")

from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import os

app = Flask(__name__)
model = YOLO("model/yolov8n.pt")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_image():
    file = request.files["image"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = model(img)
    output_img = results[0].plot()
    cv2.imwrite("static/result.jpg", output_img)
    return jsonify(count=len(results[0].boxes))
print("Сохраняю результат в static/result.jpg")

if __name__ == "__main__":
    app.run(debug=True)
