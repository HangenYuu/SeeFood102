import lightning as L
import torch
import timm
import hydra

import torchvision.transforms as T
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from food101classifier import Food101Classifier, Food101DataModule
torch.set_float32_matmul_precision('high')

@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg):
    logger = TensorBoardLogger("runs", name=f"{cfg.model.name}/logs")
    food_model = Food101Classifier("hf_hub:timm/" + cfg.model.name + ".fb_dist_in1k")
    data_cfg = timm.data.resolve_data_config(food_model.model.pretrained_cfg) # type: ignore
    test_transform = timm.data.create_transform(**data_cfg) # type: ignore
    train_transform = T.Compose([
                    T.Resize(248),
                    T.RandomAffine(scale=(0.9, 1.1), translate=(0.1, 0.1), degrees=10),
                    T.RandomHorizontalFlip(0.5),
                    T.RandomCrop(224, padding_mode="reflect", pad_if_needed=True),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
    food_data = Food101DataModule(train_transform, test_transform, batch_size = cfg.processing.batch_size)
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
    trainer.test(food_model, food_data)

if __name__ == "__main__":
    main()