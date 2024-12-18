from typing import Literal

Optimizer = Literal["Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"]
FoldType = Literal["train", "val", "test"]
IntegrationType = Literal["wandb", "mlflow", "tensorboard"]
