from argparse import ArgumentParser

def add_arguments(parser: ArgumentParser) -> None:
    
    # Run args
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="WandB group name for organizing runs.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name of the run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default="",
        help="Comma-separated list of tags for the WandB run.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to use WandB for logging.",
    )
    
    # Data args
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="/dtu/blackhole/10/178320/preprocessed_1/final",
        help="Path to the preprocessed dataset directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default='data/checkpoints',
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=64,
        help="Input sequence length for training.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Prediction horizon.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="LAT,LON,SOG,COG,HEADING,ROT,NAV_STT,TIMESTAMP,MMSI",
        help="Comma-separated list of features to use.",
    )
    
    # Model args
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["gru", "tptrans", "traisformer"],
        default="tptrans",
        help="Type of model to use",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=192,
        help="Dimension of the model.",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=4,
        help="Number of attention heads (for TPTrans).",
    )
    parser.add_argument(
        "--enc_layers",
        type=int,
        default=4,
        help="Number of encoder layers (for TPTrans).",
    )
    parser.add_argument(
        "--dec_layers",
        type=int,
        default=2,
        help="Number of decoder layers (for TPTrans).",
    )
    
    # Training args
    parser.add_argument(
        "--loss",
        type=str,
        choices=["mse", "huber"],
        default="huber",
        help="Loss function to use: 'mse' or 'huber'.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw"],
        default="adamw",
        help="Optimizer to use.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=96,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )