import lightning as L
import torch
import timm
import hydra

from torchvision import transforms
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from food101classifier import Food101Classifier, Food101DataModule
torch.set_float32_matmul_precision('high')

@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg):
    logger = TensorBoardLogger("runs", name=f"{cfg.model.name}/logs")
    food_model = Food101Classifier("hf_hub:timm/" + cfg.model.name + ".fb_dist_in1k")
    transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
    food_data = Food101DataModule(transform, batch_size = cfg.processing.batch_size)
    checkpoint_callback = ModelCheckpoint(monitor="val_acc",
                                          dirpath="models",
                                          filename=f"{cfg.model.name}/checkpoints",
                                          mode = 'max')
    trainer = L.Trainer(
        logger=logger, # type: ignore
        accelerator='gpu',
        devices=2,
        precision="16-mixed",
        accumulate_grad_batches=cfg.training.max_epochs,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        max_epochs=cfg.training.accumulate_grad_batches,
        fast_dev_run=False,
        profiler="advanced",
        log_every_n_steps=cfg.training.log_every_n_steps
    )
    trainer.fit(food_model, food_data)

if __name__ == "__main__":
    main()