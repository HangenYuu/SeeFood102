from fastapi import FastAPI, UploadFile, File
from inference_onnx import Food101ONNXPredictor
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import io

app = FastAPI(title="Food102 (Food101 + MLOps) App")

origins = [
    "http://localhost",
    "http://localhost:5500",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://0.0.0.0",
    "http://0.0.0.0:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = Food101ONNXPredictor("./models/levit_256/onnx/checkpoints.onnx")

@app.get("/")
async def home_page():
    return """Sample prediction API
    If you are seeing this, it means that the app is running.
    Navigate to <The URL you are seeing>/predict to get a prediction."""

@app.post("/predict")
async def get_prediction(image: UploadFile = File(...)):
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    result = predictor.predict(pil_image)
    return result