import json
import base64
from inference_onnx import Food101ONNXPredictor

inferencing_instance = Food101ONNXPredictor("models/levit_256/onnx/checkpoints.onnx")

def lambda_handler(event, context):
    """
    Lambda function handler for predicting linguistic acceptability of the given sentence
    """

    encoded_image = event["image"]
    decoded_image = base64.b64decode(encoded_image)
    result = inferencing_instance.predict(decoded_image)

    return {
        "statusCode": 200,
        "body": json.dumps(result)
    }