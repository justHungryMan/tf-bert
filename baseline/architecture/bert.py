import tensorflow as tf
import tensorflow_addons as tfa
import math
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
)

class MaskLM(tf.keras.layers.Layer):
    def __init__(self, vocab_size, num_hiddens, d_model, **kwargs):
        super(MaskLM, self).__init__()
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units=num_hiddens,
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                bias_initializer="zeros",
                activation='relu'
            ),
            LayerNormalization(epsilon=1e-6),
            tf.keras.layers.Dense(
                units=vocab_size,
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                bias_initializer="zeros",
            )
        ])
    
    def call(self, x, masked_lm_positions):
        x = tf.gather(x, masked_lm_positions, batch_dims=1)
        x = self.mlp(x)

        return x

class NspLM(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NspLM, self).__init__()

        self.linear =  tf.keras.layers.Dense(
            units=2,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            bias_initializer="zeros",
            activation='relu'
        )

    def call(self, x):
        x = self.linear(x)

        return x

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8, attn_drop_rate=0.0, proj_drop=0.0):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads

        self.query_dense = tf.keras.layers.Dense(
            units=embed_dim,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            bias_initializer="zeros",
        )
        self.key_dense = tf.keras.layers.Dense(
            units=embed_dim,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            bias_initializer="zeros",
        )
        self.attn_drop = tf.keras.layers.Dropout(rate=attn_drop_rate)
        self.value_dense = tf.keras.layers.Dense(
            units=embed_dim,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            bias_initializer="zeros",
        )
        self.combine_heads = tf.keras.layers.Dense(
            units=embed_dim,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            bias_initializer="zeros",
        )
        self.proj_drop = tf.keras.layers.Dropout(rate=proj_drop)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        weights = self.attn_drop(weights)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        output = self.proj_drop(output, training=training)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "embed_dim": self.embed_dim,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_dim,
        drop_rate,
        attn_drop_rate,
        name="encoderblock",
    ):
        super(TransformerBlock, self).__init__(name=name)

        self.att = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop_rate=attn_drop_rate,
            proj_drop=drop_rate,
        )

        self.mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    units=mlp_dim,
                    activation="linear",
                    kernel_initializer=tf.keras.initializers.GlorotNormal(),
                    bias_initializer=tf.keras.initializers.RandomNormal(
                        mean=0.0, stddev=1e-6
                    ),
                ),
                tf.keras.layers.Lambda(
                    lambda x: tf.keras.activations.gelu(x, approximate=False)
                ),
                tf.keras.layers.Dropout(rate=drop_rate),
                tf.keras.layers.Dense(
                    units=embed_dim,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(),
                    bias_initializer=tf.keras.initializers.RandomNormal(
                        mean=0.0, stddev=1e-6
                    ),
                ),
                tf.keras.layers.Dropout(rate=drop_rate),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.att(inputs_norm, training=training)
        out1 = attn_output + inputs

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        return out1 + mlp_output

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Bert(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        num_classes,
        d_model,
        num_heads,
        mlp_dim,
        drop_rate=0.1,
        attn_drop_rate=0.0,
    ):
        super(Bert, self).__init__()

        self.mlm_loss_tracker = tf.keras.metrics.Mean(name="mlm_loss")
        self.nsp_loss_tracker = tf.keras.metrics.Mean(name="nsp_loss")
        self.mlm_acc_tracker = tf.keras.metrics.Accuracy(name="mlm_acc")
        self.nsp_acc_tracker = tf.keras.metrics.Accuracy(name="nsp_acc")

        self.d_model = d_model
        self.num_layers = num_layers


        self.embedding = tf.keras.layers.Embedding(
            input_dim=30_522,
            output_dim=d_model,
            name="token_embedding",
        )

        self.segment_embedding = tf.keras.layers.Embedding(
            input_dim=2,
            output_dim=d_model,
            name="segment_embedding",
        )

        self.pos_emb = self.add_weight(
            "pos_emb",
            shape=(1, 128, d_model),
            initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=0.02, seed=None
            ),
            trainable=True,
        )

        self.pos_drop = tf.keras.layers.Dropout(rate=drop_rate, name="pos_drop")


        self.enc_layers = [
            TransformerBlock(
                embed_dim=d_model,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                name=f"encoderblock_{i}",
            )
            for i in range(num_layers)
        ]

        self.norm = LayerNormalization(epsilon=1e-6, name="encoder_norm")

        self.mlm = MaskLM(
            vocab_size=30_522, 
            num_hiddens=128,
            d_model=d_model,
            name="maskLM"
        )
        self.hidden = tf.keras.layers.Dense(
            units=128,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            bias_initializer="zeros",
            activation='tanh',
            name="hidden"
        )
        self.nsplm = NspLM(
            name="nspLM"
        )


    def train_step(self, data):
        input_ids = data["input_ids"]
        masked_lm_ids = data["masked_lm_ids"]
        masked_lm_positions = data["masked_lm_positions"]
        masked_lm_weights = data["masked_lm_weights"]
        next_sentence_labels = data["next_sentence_labels"]
        segment_ids = data["segment_ids"]

        input_data = {
            "input_ids": input_ids,
            "masked_lm_positions": masked_lm_positions,
            "segment_ids": segment_ids
        }
        with tf.GradientTape() as tape:
            mlm_y_hat, nsp_y_hat = self(input_data, training=True)

            mlm_logits = tf.reshape(mlm_y_hat, [-1, 30_522])
            mlm_labels = tf.one_hot(
                tf.multiply(
                    tf.reshape(
                        masked_lm_ids, 
                        [-1]
                        ), 
                    tf.reshape(
                        tf.cast(
                            masked_lm_weights, 
                            tf.int32
                            ), 
                        [-1]
                        )
                    ), 
                30_522
                )
            mlm_loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=mlm_logits,
                labels=mlm_labels
            )
            
            nsp_labels = tf.squeeze(tf.one_hot(next_sentence_labels, 2), 1)
            nsp_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=nsp_y_hat, 
                labels=nsp_labels,
                )
            

            mlm_loss = tf.math.divide_no_nan(tf.math.reduce_mean(mlm_loss), tf.math.reduce_mean(masked_lm_weights))
            loss = mlm_loss + tf.reduce_sum(nsp_loss)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))


        self.mlm_loss_tracker.update_state(mlm_loss)
        self.mlm_acc_tracker.update_state(mlm_labels, mlm_logits)
        self.nsp_loss_tracker.update_state(nsp_loss)
        self.nsp_acc_tracker.update_state(nsp_labels, nsp_y_hat)
        return {
            "mlm_loss": self.mlm_loss_tracker.result(),
            "mlm_acc": self.mlm_acc_tracker.result(),
            "nsp_loss": self.nsp_loss_tracker.result(),
            "nsp_acc": self.nsp_acc_tracker.result()
            }
    @property
    def metrics(self):
        return [self.mlm_loss_tracker, self.mlm_acc_tracker, self.nsp_loss_tracker, self.nsp_acc_tracker]

    def call(self, data, training):
        input_ids = data["input_ids"]
        masked_lm_positions = tf.cast(data["masked_lm_positions"], tf.int32)
        segment_ids = data["segment_ids"]
        batch_size = input_ids.shape[0]
        
        token_embedding = self.embedding(input_ids)
        segment_embedding = self.segment_embedding(segment_ids)

        # B x (max_seq * 3) x d_model
        x = token_embedding + segment_embedding
        x += self.pos_emb
        x = self.pos_drop(x, training=training)

        for layer in self.enc_layers:
            x = layer(x, training=training)

        x = self.norm(x)

        mlm_y_hat = self.mlm(x, masked_lm_positions)

        nsp_y_hat = self.nsplm(self.hidden(x[:, 0, :]))

        return mlm_y_hat, nsp_y_hat


KNOWN_MODELS = {
    "ti": {
        "num_layers": 12,
        "d_model": 192,
        "num_heads": 3,
        "mlp_dim": 768,
    },
    "s": {
        "num_layers": 12,
        "d_model": 384,
        "num_heads": 6,
        "mlp_dim": 1536,
    },
    "b": {
        "num_layers": 12,
        "d_model": 768,
        "num_heads": 12,
        "mlp_dim": 3072,
    },
    "l": {
        "num_layers": 24,
        "d_model": 1024,
        "num_heads": 16,
        "mlp_dim": 4096,
    },
}


def create_name(arechitecture_name, num_classes, **kwargs):
    base = arechitecture_name.lower()
    return Bert(
        num_classes=num_classes,
        **KNOWN_MODELS[base],
        **kwargs,
    )
