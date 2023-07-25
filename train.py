import lightning as L
import torch
import timm
import gc


from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from food101classifier import Food101Classifier, Food101DataModule
torch.set_float32_matmul_precision('high')

def main():
    model = "levit_256.fb_dist_in1k"
    print(model)
    logger = TensorBoardLogger("runs", version=1, name=f"{model}/logs")
    food_model = Food101Classifier("hf_hub:timm/"+model)
    data_cfg = timm.data.resolve_data_config(food_model.model.pretrained_cfg) # type: ignore
    transform = timm.data.create_transform(**data_cfg) # type: ignore
    food_data = Food101DataModule(transform)
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", dirpath="models", filename=f"{model}/checkpoints")
    trainer = L.Trainer(
        logger=logger, # type: ignore
        accelerator='gpu',
        devices=2,
        precision="16-mixed",
        accumulate_grad_batches=1,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        max_epochs=20,
        fast_dev_run=False,
        profiler="advanced",
    )
    trainer.fit(food_model, food_data)

if __name__ == "__main__":
    main()