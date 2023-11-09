from xformers.factory import xFormerEncoderBlock, xFormerEncoderConfig
import torch

BATCH = 3
SEQ = 64*64*64
EMB = 128
VOCAB = 128

encoder_config = {
    "dim_model": EMB,
    "residual_norm_style": "pre",  # Optional, pre/post
    "position_encoding_config": {
        "name": "vocab",  # whatever position encodinhg makes sense
        "seq_len": SEQ,
        "vocab_size": VOCAB,
    },
    "multi_head_config": {
        "num_heads": 2,
        "residual_dropout": 0.2,
        "attention": {
            "name": "linformer",  # whatever attention mechanism
            "dropout": 0,
            "seq_len": SEQ,
        },
    },
    "feedforward_config": {
        "name": "MLP",
        "dropout": 0,
        "activation": "relu",
        "hidden_layer_multiplier": 2,
    },
}

if __name__ == "__main__":
    # "constructing" the config will lead to a lot of type checking,
    # which could catch some errors early on
    config = xFormerEncoderConfig(**encoder_config)

    encoder = xFormerEncoderBlock(config)

    total_params = sum(p.numel() for p in encoder.parameters())
    print("total custom  pretrained params", total_params)

    #  Test out with dummy inputs
    x = (torch.rand((BATCH, SEQ)) * VOCAB).abs().to(torch.int)
    y = encoder(x)
    print(y.shape)