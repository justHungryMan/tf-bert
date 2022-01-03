import tensorflow as tf

from baseline.utils import get_logger

log = get_logger(__name__)


def create(config):
    if config["type"].lower() == "ce":
        log.info(f"[loss] create CategoricalCrossEntropy")

        return tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    else:
        raise AttributeError(f"not support loss config: {config}")
