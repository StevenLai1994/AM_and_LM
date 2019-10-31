


def lm_hparams():
    params = tf.contrib.training.HParams(
        num_heads = 8,
        num_blocks = 6,
        # vocab
        input_vocab_size = 50,
        label_vocab_size = 50,
        # embedding size
        max_length = 100,
        hidden_units = 512,
        dropout_rate = 0.2,
        lr = 0.0003,
        is_training = True)
    return params