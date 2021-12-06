import os
import logging
import pickle
import datetime

# PyTorch libraries
import torch

# My libraries
from deeplearning import nn_registry, init_weights

def init_logger(config):
    """Initialize a logger object. 
    """
    log_level = config.log_level
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    fh = logging.FileHandler(config.log_file)
    fh.setLevel(log_level)
    sh = logging.StreamHandler()
    sh.setLevel(log_level)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("-"*80)

    return logger

def save_record(config, record):
    current_path = os.path.dirname(__file__)
    current_time = datetime.datetime.now()
    current_time_str = datetime.datetime.strftime(current_time ,'%H_%M')
    file_name = config.record_dir.format(current_time_str)
    with open(os.path.join(current_path, file_name), "wb") as fp:
        pickle.dump(record, fp)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_model(config, logger):
    logger.info("Initialize the full precision model.")
    # initialize the full precision model
    sample_size = config.sample_size[0] * config.sample_size[1] * config.channels
    model = nn_registry[config.model](in_dims=sample_size, in_channels=config.channels, out_dims=config.classes)
    model.apply(init_weights)

    if os.path.exists(config.full_weight_dir):
        logger.info("--- Load pre-defined full precision model. ---")
        state_dict = torch.load(config.full_weight_dir)
        model.load_state_dict(state_dict)

    model.to(config.device)

    return model

def init_record(config, model):
    record = {}
    # number of trainable parameters
    record["num_parameters"] = count_parameters(model)

    # initialize data record 
    record["testing_accuracy"] = []
    record["loss"] = []

    return record

def parse_dataset_type(config):
    if "fmnist" in config.train_data_dir:
        type_ = "fmnist"
    elif "mnist" in config.train_data_dir:
        type_ = "mnist"
    elif "cifar" in config.train_data_dir:
        type_ = "cifar"
    
    return type_