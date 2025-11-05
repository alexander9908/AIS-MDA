from argparse import ArgumentParser, Namespace
import wandb
import os
from wandb.sdk.wandb_run import Run as wandb_run
import traceback

def main(args: Namespace, logger: wandb_run):

    print("Starting preprocessing...")
    
    # Create pickle dir
    os.makedirs(os.path.join(args.output_dir, 'tmp_pickle_files'), exist_ok=True)
    args.tmp_pickle_dir = os.path.join(args.output_dir, 'tmp_pickle_files')
    
    logger.config.update(vars(args))
    
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Preprocess AIS data")
    parser.add_argument("-input_dir", type=str, required=True, help="Input directory for raw AIS data")
    parser.add_argument("-output_dir", type=str, required=True, help="Output directory for preprocessed data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--test", action="store_true", help="Run in test mode with limited data")
    parser.add_argument("--lat_min", type=float, default=54.0, help="Minimum latitude for bounding box")
    parser.add_argument("--lat_max", type=float, default=59.0, help="Maximum latitude for bounding box")
    parser.add_argument("--lon_min", type=float, default=5.0, help="Minimum longitude for bounding box")
    parser.add_argument("--lon_max", type=float, default=17.0, help="Maximum longitude for bounding box")
    parser.add_argument("--sog_max", type=float, default=30.0, help="Maximum speed over ground (SOG) in knots")
    parser.add_argument("--duration_max", type=int, default=24, help="Maximum duration of voyages in hours")
    
    args = parser.parse_args()
    
    logger = None
    try:
        logger = wandb.init(
                project="AIS-MDA",
                group="Preprocessing",
                job_type="preprocessing"
            )
        main(args, logger)
    except Exception as e:
        print(f"Script failed with exception: {e}")
        traceback.print_exc()
        if logger is not None:
            logger.log({"error": str(e)})
            logger.log({"traceback": traceback.format_exc()})
            logger.finish(exit_code=1)
    
    finally:
        if logger:
            logger.finish(exit_code=0)