from PIL import Image
import torch
import torchvision.transforms as T
from food101classifier import Food101Classifier

class Food101Predictor:
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        if torch.cuda.is_available():
            self.model = Food101Classifier.load_from_checkpoint(model_path) # type: ignore
        else:
            self.model = Food101Classifier.load_from_checkpoint(model_path, map_location='cpu')
        self.model.eval()
        self.model.freeze()
        self.transform = T.Compose([
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        self.softmax = torch.nn.Softmax(dim=0)
        with open('labels.txt', 'r') as f:
            self.idx_to_label = [s.strip() for s in f.readlines()]
    
    def predict(self, input_image):
        input_tensor = self.transform(input_image)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            print('CUDA device detected. Switched to CUDA device for faster inference')
            input_batch = input_batch.to('cuda')
        else:
            print('Using CPU for inference. Will be slower')
        
        with torch.inference_mode():
            output = self.model(input_batch)
        
        probabilities = self.softmax(output[0])
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        # Label:probability
        result = {self.idx_to_label[int(idx)]:val.item() for val, idx in zip(top5_prob.cpu(), top5_catid.cpu())}
        return result
    
if __name__ == "__main__":
    image_path = "pablo-pacheco-D3Mag4BKqns-unsplash.jpg"
    pil_image = Image.open(image_path)
    predictor = Food101Predictor("models/levit_256/checkpoints.ckpt")
    print(predictor.predict(pil_image))