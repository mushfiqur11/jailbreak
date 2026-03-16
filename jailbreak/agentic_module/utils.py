from datasets import load_from_disk
import logging

def load_jailbreak_dataset(config):
    if config.data_dir:
        try:
            ds = load_from_disk(config.data_dir)
            logging.info(f"Loaded dataset from disk at {config.data_dir}")
        except Exception as e:
            from datasets import load_dataset
            if config.dataset_name is None:
                config.dataset_name = "JailbreakBench/JBB-Behaviors"
            ds = load_dataset(config.dataset_name, "behaviors")
            ds.save_to_disk(config.data_dir)
            logging.info(f"Downloaded and saved dataset to {config.data_dir}")
    else:
        logging.warning("No data path provided")
    return ds
    