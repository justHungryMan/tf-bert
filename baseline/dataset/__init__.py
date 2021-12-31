import tensorflow as tf

from .tf_records import Bert_dataset


def create(conf_dataset, seed=None, num_devices=None):
    if conf_dataset["type"] == "tensorflow_records":
        train_config = conf_dataset["train"]
        info = {
            "num_examples": 74004228 * 8,
            "num_classes": 30_522,
            "num_shards": 1024
        }
        return {
            "train": Bert_dataset(
                train_config,
                data_dir=conf_dataset["data_dir"],
                seed=seed,
                num_devices=num_devices,
            ),
            "train_info": info
        }
    else:
        raise AttributeError(f"not support dataset/type config: {config}")
