from argparse import ArgumentParser
from omegaconf import OmegaConf
from jailbreak.agentic_module.utils import load_jailbreak_dataset
import logging

def main(config):
    ds = load_jailbreak_dataset(config.data)
    logging.info(ds)

    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the configuration file"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    logging.basicConfig(level=logging.INFO)
    main(config)
    
