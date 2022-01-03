import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import os
import json
import logging

LOGGER = logging.getLogger(__name__)

class Bert_dataset:
    def __init__(self, config, data_dir, seed, num_devices):
        self.conf = config
        self.data_dir = data_dir
        self.seed = seed
        self.num_devices = num_devices
        self.files = tf.io.gfile.glob(os.path.join(self.data_dir, '*'))
    
    def get_input_fn(self):
        logging.info(f'Data Directory: {self.data_dir}, num_files: {len(self.files)}')

        def parse_record(record):
            keys_to_features = {
                "input_ids": tf.io.FixedLenFeature([128], tf.int64),
                "segment_ids": tf.io.FixedLenFeature([128], tf.int64),
                "masked_lm_positions": tf.io.FixedLenFeature([20], tf.int64),
                "masked_lm_ids": tf.io.FixedLenFeature([20], tf.int64),
                "masked_lm_weights": tf.io.FixedLenFeature([20], tf.float32),
                "next_sentence_labels": tf.io.FixedLenFeature([1], tf.int64),  
            }
            example = tf.io.parse_single_example(record, keys_to_features)
            
            # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
            # So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.cast(t, tf.int32)
                example[name] = t

            return example
    
        def input_fn(input_context):
            ds = tf.data.Dataset.from_tensor_slices(self.files)
            buffer_size = 8 * 1024 * 1024

            ds = ds.interleave(lambda name: tf.data.TFRecordDataset(name, buffer_size=buffer_size),
                                cycle_length=16, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            
            local_batch_size = input_context.get_per_replica_batch_size(self.conf.batch_size)

            if input_context.num_input_pipelines > 1:
                ds = ds.cache()
            
            ds = ds.shuffle(16 * local_batch_size, seed=int(self.seed, 0)).repeat()

            ds = ds.map(parse_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            ds = ds.batch(batch_size=local_batch_size, drop_remainder=True)
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

            # from deepmind's code : https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L91
            options = tf.data.Options()
            options.experimental_deterministic = False
            options.experimental_threading.private_threadpool_size = 48
            options.experimental_threading.max_intra_op_parallelism = 1
            ds = ds.with_options(options)


            return ds
        
        return input_fn