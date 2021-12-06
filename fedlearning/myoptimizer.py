import numpy as np
import logging
import copy

# PyTorch libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

# My libraries
from config.utils import parse_dataset_type
from fedlearning import predictor_registry, quantizer_registry, encoder_registry
from fedlearning.buffer import *
from deeplearning import UserDataset

class LocalUpdater(object):
    def __init__(self, user_resource, config, **kwargs):
        """Construct a local updater for a user.

        Args:
            user_resources(dict):   a dictionary containing images and labels listed as follows. 
                - images (ndarray):  training images of the user.
                - labels (ndarray): training labels of the user.

            config (class):         global configuration containing info listed as follows:
                - lr (float):       learning rate for the user.
                - batch_size (int): batch size for the user. 
                - device (str):     set 'cuda' or 'cpu' for the user. 
                - predictor (str):  predictor type.
                - quantizer (str):  quantizer type.
        """
        
        try:
            self.lr = user_resource["lr"]
            self.batch_size = user_resource["batch_size"]
            self.device = user_resource["device"]
            
            assert("images" in user_resource)
            assert("labels" in user_resource)
        except KeyError:
            logging.error("LocalUpdater Initialization Failure! Input should include `lr`, `batch_size`!") 
        except AssertionError:
            logging.error("LocalUpdater Initialization Failure! Input should include samples!") 

        self.local_weight = None
        self.init_weight = None
        dataset_type = parse_dataset_type(config)

        self.sample_loader = \
            DataLoader(UserDataset(user_resource["images"], 
                        user_resource["labels"],
                        dataset_type), 
                sampler=None, 
                batch_size=self.batch_size,
                shuffle=True)

        self.criterion = nn.CrossEntropyLoss()
 
        self.quantizer = quantizer_registry[config.quantizer](config)

        if config.backup_quantizer in quantizer_registry.keys():
            self.baseline_quantizer = quantizer_registry[config.backup_quantizer](config)
        else:
            self.baseline_quantizer = None

        self.entropy_encoder = encoder_registry[config.entropy_encoder](config)
        self.tau = config.tau
        
        self.lambda_ = config.lambda_
        self.total_codewords = config.quantization_level

    def local_step(self, model):
        """Perform local update tau times.

        Args,
            model(nn.module):       the global model.
            offset(tensor):         delta offset term preventing client drift.
        """
        self.init_weight = copy.deepcopy(model.state_dict())
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        
        tau_counter = 0
        break_flag = False

        while not break_flag:
            for sample in self.sample_loader:
                image = sample["image"].to(self.device)
                label = sample["label"].to(self.device)
                optimizer.zero_grad()

                output = model(image)
                loss = self.criterion(output, label)
                loss.backward()
                optimizer.step()                              # w^(c+1) = w^(c) - \eta \hat{grad}

                tau_counter += 1
                if tau_counter >= self.tau:
                    break_flag = True
                    break
        
        self.local_weight = copy.deepcopy(model.state_dict())
        
        # load back the model copy hence the global model won't be changed
        model.load_state_dict(self.init_weight)

    def uplink_transmit(self, predicted_weights):
        """Simulate the transmission of residual between local weight and predicted weight.
        Predictor is designed to be same on local workers and the global server, so predicted_weight
        can simply be a copy from global predicted version.

        Args,
            predicted_weight(OrderedDict):   state_dict of the predicted weight.
        """ 

        try:
            assert(self.local_weight != None)
        except AssertionError:
            logging.error("No local model buffered!")

        # calculate the weight residual, then quantize and compress
        quantized_sets = {}
        coded_sets = {}
        pred_modes = {}
        quant_modes = {}

        for i, w_name in enumerate(self.init_weight):
            residuals = []
            rmses = []
            for j in range(len(predicted_weights)):
                residual = self.local_weight[w_name] - predicted_weights[j][w_name]
                residuals.append(residual)
                
                rmse = (residual.norm()).item()*(1/np.sqrt(residual.numel()))
                rmses.append(rmse)
 
            mode = np.argmin(rmses)
            pred_modes[w_name] = mode
            rmse = rmses[mode]
            residual = residuals[mode]

            quantized_set = self.quantizer.quantize(residual)
            dequantized_signal = self.quantizer.dequantize(quantized_set)
            normal_quantized_err = torch.norm(residual - dequantized_signal)*(1/np.sqrt(residual.numel()))
            normal_RD_cost = normal_quantized_err + self.lambda_*self._entropy(quantized_set["quantized_arr"])

            if self.baseline_quantizer is not None:
                baseline_quantized_set = self.baseline_quantizer.quantize(residual)
                baseline_dequantized_signal = self.baseline_quantizer.dequantize(baseline_quantized_set)
                baseline_quantized_err = torch.norm(residual - baseline_dequantized_signal)*(1/np.sqrt(residual.numel()))
                baseline_RD_cost = baseline_quantized_err + self.lambda_*self._entropy(baseline_quantized_set["quantized_arr"])

                if normal_RD_cost > baseline_RD_cost:
                    quant_modes[w_name] = "baseline"
                    quantized_set = baseline_quantized_set
                else:
                    quant_modes[w_name] = "normal"
                    quantized_set = quantized_set
            else:
                # fix to baseline quantizer:
                quant_modes[w_name] = "normal"
                quantized_set = quantized_set

            coded_set = self.entropy_encoder.encode(quantized_set["quantized_arr"]) 

            quantized_sets[w_name] = quantized_set
            coded_sets[w_name] = coded_set

        local_package = dict(quantized_sets=quantized_sets, 
                             coded_sets=coded_sets,
                             pred_modes=pred_modes,
                             quant_modes=quant_modes)

        return local_package

    def _entropy(self, seq):
        histogram = torch.histc(seq, bins=self.total_codewords, min=0, max=self.total_codewords-1) 
        total_symbols = seq.numel()

        histogram = histogram.detach().cpu().numpy().astype("float")
        histogram /= total_symbols

        entropy = 0
        for i, prob in enumerate(histogram):
            if prob == 0:
                continue
            entropy += -prob * np.log2(prob)

        return entropy


class GlobalUpdater(object):
    def __init__(self, config, initial_model, **kwargs):
        """Construct a global updater for a server.

        Args:
            config (class):              global configuration containing info listed as follows:
                - predictor (str):       predictor type.
                - quantizer (str):       quantizer type.
                - entropy_encoder (str): entropy encoder type.

            initial_model (OrderedDict): initial model state_dict
        """
        # predictor_type indicates whether the predictor outputs 
        # accumulated gradients or model weights
        self.predictors = []
        self.predictor_types = []   
        for predictor_name in config.predictor:
            predictor = predictor_registry[predictor_name](config)
            if "adaM" in predictor_name or "delta" in predictor_name:
                self.predictor_types.append("accgrad")
                predictor.init_delta_buffers(initial_model)
            else:
                self.predictor_types.append("weight")
                predictor.init_weight_buffers(initial_model)

            self.predictors.append(predictor)

        self.num_predictors = len(config.predictor)
        self.predicted_weights = []
        for i in range(self.num_predictors):
            self.predicted_weights.append(copy.deepcopy(initial_model))

        self.quantizer = quantizer_registry[config.quantizer](config)

        self.entropy_encoder = encoder_registry[config.entropy_encoder](config)

        self.num_users = int(config.users * config.part_rate)

    def global_step(self, model, local_packages, **kwargs):
        """Perform a global update with collocted coded info from local users.
        """
        avg_weight = WeightBuffer(self.predicted_weights[0], mode="zeros")
        avg_weight_dict = avg_weight.state_dict()

        for user_id, package in local_packages.items():
            
            coded_sets = package["coded_sets"]
            quantized_sets = package["quantized_sets"]
            pred_modes = package["pred_modes"]
            quant_modes = package["quant_modes"]
            
            for w_name, coded_set in coded_sets.items():

                # decode
                decoded_quantized_residual = self.entropy_encoder.decode(coded_set)              
                decoded_quantized_residual = decoded_quantized_residual.view(self.predicted_weights[0][w_name].shape)
                decoded_quantized_residual = decoded_quantized_residual.to(self.predicted_weights[0][w_name])
  
                # dequantize
                quantized_sets[w_name]["quantized_arr"] = decoded_quantized_residual 
                residual = self.quantizer.dequantize(quantized_sets[w_name])

                mode = pred_modes[w_name]
                avg_weight_dict[w_name] += self.predicted_weights[mode][w_name] + residual

        avg_weight.push(avg_weight_dict)
        avg_weight *= (1/self.num_users)
        avg_weight_dict = avg_weight.state_dict()  
        
        delta_weight_dict = {}
        model_weight_dict = model.state_dict()

        for w_name, w_value in avg_weight_dict.items():
            delta_weight_dict[w_name] = model_weight_dict[w_name] - w_value 

        for i in range(self.num_predictors):
            if self.predictor_types[i] == "weight":
                self.predictors[i].update_buffer(avg_weight_dict)
            else:
                self.predictors[i].update_buffer(delta_weight_dict)

        model.load_state_dict(avg_weight_dict)

    def predict_weight(self, model, predictor_index):
        self.predicted_weights[predictor_index] = self.predictors[predictor_index].predict(model.state_dict())
        return self.predicted_weights[predictor_index]
