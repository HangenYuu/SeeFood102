from food101classifier import Food101Classifier
import logging
import hydra
import torch

logger = logging.getLogger(__name__)

def convert_model(model_checkpoint):
    root_dir = hydra.utils.get_original_cwd()
    model_path = f"{root_dir}/models/levit_256/{model_checkpoint}.ckpt"
    
    logger.info(f"Loading pre-trained model from: {model_path}")
    model = Food101Classifier.load_from_checkpoint(model_path, map_location='cpu')
    input_sample = torch.randn((1, 3, 224, 224))
    filepath = f"{root_dir}/models/levit_256/onnx/{model_checkpoint}.onnx"
    
    logger.info(f"Converting the model into ONNX format")
    model.to_onnx(filepath,
              input_sample,
              export_params=True,
              input_names = ['input'],    # Input names
              output_names = ['output'],  # Output names
              dynamic_axes={              # variable length axes
                'input' : {0 : 'batch_size'},
                'output' : {0 : 'batch_size'},
                }
              )
    logger.info(
        f"Model converted successfully. ONNX format model is at: {root_dir}/models/model.onnx"
    )

if __name__ == "__main__":
    convert_model("checkpoints")