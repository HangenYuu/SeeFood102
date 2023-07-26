from PIL import Image
import torch
import torchvision.transforms as T
from utils import timing
import onnxruntime as ort

class Food101ONNXPredictor:
    def __init__(self, model_path) -> None:
        self.ort_session = ort.InferenceSession(model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.transform = T.Compose([
                    T.Resize(248),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        self.softmax = torch.nn.Softmax(dim=0)
        with open('labels.txt', 'r') as f:
            self.idx_to_label = [s.strip() for s in f.readlines()]

    @timing
    def predict(self, input_image):
        input_tensor = self.transform(input_image)
        input_batch = input_tensor.unsqueeze(0).numpy()
        ort_inputs = {self.input_name: input_batch}
        ort_outs = torch.Tensor(self.ort_session.run(None, ort_inputs)[0])
        probabilities = self.softmax(ort_outs[0])
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        # Label:probability
        result = {self.idx_to_label[int(idx)]:val.item() for val, idx in zip(top5_prob.cpu(), top5_catid.cpu())}
        return result


if __name__ == "__main__":
    image_path = "pablo-pacheco-D3Mag4BKqns-unsplash.jpg"
    pil_image = Image.open(image_path)
    predictor = Food101ONNXPredictor("models/levit_256/onnx/checkpoints-v1.onnx")
    print(predictor.predict(pil_image))
    # for _ in range(20):
    #     predictor.predict(pil_image)