import time
import numpy as np

# PyTorch libraries
import torch

# My libraries
from config import load_config
from config.utils import *
from fedlearning.myoptimizer import *
from deeplearning.dataset import *
from deeplearning.validate import *

def train(model, config, logger, record):
    """Simulate Federated Learning training process. 
    
    Args:
        model (nn.Module):       the model to be trained.
        config (class):          the user defined configuration.
        logger (logging.logger): a logger for train info output.
        record (dict):           a record for train info saving.  
    """    
    num_predictors = len(config.predictor)

    # initialize user_ids
    user_ids = np.arange(int(config.users*config.part_rate))

    # initialize the optimizer for the server model
    dataset = assign_user_data(config, logger)

    # global_updater = GlobalUpdater(config, model.state_dict())
    global_updater = GlobalUpdater(config, model.state_dict(), 
                                   dim=record["num_parameters"], 
                                   device=config.device) 

    dataset_type = parse_dataset_type(config)
    # before optimization, report the result first
    with torch.no_grad():
        # validate the model and log test accuracy
        loss = train_loss(model, dataset["train_data"], dataset_type, device=config.device)
        test_acc = test_accuracy(model, dataset["test_data"], dataset_type, device=config.device)
        
        record["loss"].append(loss)
        record["testing_accuracy"].append(test_acc)

        logger.info("Test accuracy {:.4f}".format(test_acc))
        logger.info("Train loss {:.4f}".format(loss))
        logger.info("")
    
    for comm_round in range(config.rounds):

        # The server predict the weight in the next step and wait for 
        # users' residual between local weight and the predcited version
        logger.info("Round {:d}".format(comm_round))
        
        predicted_weights = []
        for predictor_index in range(num_predictors):
            predicted_weights.append(global_updater.predict_weight(model, predictor_index))

        # Sample a fraction of users randomly
        user_ids_candidates = user_ids

        # Wait for all users updating locally
        local_packages = {}
        for i, userID in enumerate(user_ids_candidates):
            user_resource = assign_user_resource(config, userID, 
                                dataset["train_data"], dataset["user_with_data"])
            updater = LocalUpdater(user_resource, config)
            updater.local_step(model)
            local_package = updater.uplink_transmit(predicted_weights)

            local_packages[userID] = local_package

        # Update the global model
        global_updater.global_step(model, local_packages, record=record, logger=logger)

        # log and record
        validate_and_log(model, global_updater, dataset, config, record, logger)

        if comm_round == config.scheduler[0]:
            if "a0.5" in config.user_with_data:
                config.lr *= config.lr_scaler
            config.scheduler.pop(0)

def main(config_filename):
    config = load_config(config_filename)
    logger = init_logger(config)

    if config.device == "cuda":
        torch.backends.cudnn.benchmark = True

    model = init_model(config, logger)
    record = init_record(config, model)

    start = time.time()
    train(model, config, logger, record)
    end = time.time()
    save_record(config, record)

    logger.info("{:.3} min has elapsed".format((end-start)/60))

if __name__ == "__main__":
    main("config.yaml")

