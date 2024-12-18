import argparse

from image_similarity_search.core.callbacks import DEFAULT_CALLBACKS, add_integration
from image_similarity_search.core.train import early_stopping, save_checkpoint
from image_similarity_search.models import MODELS
from image_similarity_search.utils import IterableSimpleNamespace, yaml_load

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="CNN/AE")
    parser.add_argument("--trainset_path", type=str, default="data/processed/train.csv")
    parser.add_argument("--testset_path", type=str, default="data/processed/test.csv")
    parser.add_argument("--image_path", type=str, default="data/interim/images/images")
    parser.add_argument("--save_path", type=str, default="models/model")
    parser.add_argument(
        "--config", type=str, default="image_similarity_search/models/cnn/params.yml"
    )
    parser.add_argument(
        "--log", type=str, choices=["mlflow"], default=None, help="Track experiment"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to train the model on")
    args = parser.parse_args()

    data = {
        "path": args.image_path,
        "train": args.trainset_path,
        "val": args.testset_path,
    }

    callbacks = DEFAULT_CALLBACKS
    hyp = IterableSimpleNamespace(**yaml_load(args.config))
    # Example of adding custom callbacks
    # state.callbacks['on_train_epoch_end'].append(
    #   lambda state, *args, **kargs: print('Epoch ended')
    # )

    callbacks.register_action("on_model_save", name="save_checkpoint", callback=save_checkpoint)
    callbacks.register_action("on_val_end", name="early_stopping", callback=early_stopping)

    if args.log is not None and args.log == "mlflow":
        add_integration(callbacks, "mlflow")

    assert args.model_name in MODELS, f"Model {args.model_name} not found in {MODELS.keys()}"

    model = MODELS[args.model_name]["build"](hyp, data)
    MODELS[args.model_name]["train"](model, data, hyp, args.device, args.save_path, callbacks)
