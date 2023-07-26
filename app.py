from fastapi import FastAPI
from inference_onnx import Food101ONNXPredictor
app = FastAPI(title="Food102 (Food101 + MLOps) App")
from PIL import Image

predictor = Food101ONNXPredictor("./models/levit_256/onnx/checkpoints.onnx")

@app.get("/")
async def home_page():
    return "<h2>Sample prediction API</h2>"


@app.get("/predict")
async def get_prediction(image_path: str):
    pil_image = Image.open(image_path)
    result = predictor.predict(pil_image)
    return result