import logging

import tensorflow as tf


from baseline.utils import get_logger
from . import bert

log = get_logger(__name__)


def create(conf, num_classes=1000):
    base, architecture_name = [l.lower() for l in conf["type"].split("/")]

    if base == "bert":
        architecture = bert.create_name(
            architecture_name, num_classes=num_classes, **conf["params"]
        )
    else:
        raise AttributeError(f"not support architecture config: {conf}")

    log.info(f"[Model] create {conf['type']}")
    return architecture
