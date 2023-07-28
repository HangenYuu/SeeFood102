from fastapi import FastAPI, UploadFile, File
from inference_onnx import Food101ONNXPredictor
from PIL import Image
import io

app = FastAPI(title="Food102 (Food101 + MLOps) App")

predictor = Food101ONNXPredictor("./models/levit_256/onnx/checkpoints.onnx")

@app.get("/")
async def home_page():
    return "<h2>Sample prediction API</h2>"

@app.post("/predict")
async def get_prediction(image: UploadFile = File(...)):
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    result = predictor.predict(pil_image)
    return result