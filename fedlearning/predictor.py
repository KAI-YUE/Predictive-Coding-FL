import copy
import numpy as np

# PyTorch Libraries
import torch

# My libraries
from fedlearning import Predictor
from fedlearning.buffer import *
from deeplearning import nn_registry

class PrevFramePredictor(Predictor):
    def __init__(self, config):
        pass
    
    def init_weight_buffers(self, weight_dict):
        pass

    def predict(self, weight_dict):
        """:math: \hat{w}^{(t+1)} = w^{(t)} + \frac{1}{K} \sum_{k=1}^K \delta^{(t-k)}
        Use the history weight delta to predict the next step.
        """ 
        pred_weight_dict = copy.deepcopy(weight_dict)
        return pred_weight_dict

    def update_buffer(self, delta_weight_dict):
        pass


class OuAdaptivePredictor(Predictor):
    def __init__(self, config):
        self._weight_buffer = []
        self.step_size = config.step_size
        self.scaler = config.scaler
        self.device = config.device
    
    def init_weight_buffers(self, weight_dict):
        for w_name, w_value in weight_dict.items():
            weight_buffer = OuAdaptiveBuffer(step_size=self.step_size, scaler=self.scaler)
            weight_buffer.init_buffer_(w_value, device=self.device)
            self._weight_buffer.append(weight_buffer)

        self.buffer_len = len(self._weight_buffer)

    def predict(self, weight_dict):
        """:math: \hat{w}^{(t+1)} = w^{(t)} + \frac{1}{K} \sum_{k=1}^K \delta^{(t-k)}
        Use the history weight delta to predict the next step.
        """ 
        pred_weight_dict = copy.deepcopy(weight_dict)

        for i, w_name in enumerate(weight_dict):
            weight = weight_dict[w_name]
            pred_weight_dict[w_name] = self._weight_buffer[i].output()

        return pred_weight_dict

    def update_buffer(self, weight_dict):
        for i, w_name in enumerate(weight_dict):
            weight = weight_dict[w_name]
            self._weight_buffer[i].push(weight)
            self._weight_buffer[i].optimizer_step()


class OuArPredictor(Predictor):
    def __init__(self, config):
        self._weight_buffer = []
        self.device = config.device
    
    def init_weight_buffers(self, weight_dict):
        for w_name, w_value in weight_dict.items():
            weight_buffer = OuArBuffer()
            weight_buffer.init_buffer_(w_value, device=self.device)
            self._weight_buffer.append(weight_buffer)

        self.buffer_len = len(self._weight_buffer)

    def predict(self, weight_dict):
        """:math: \hat{w}^{(t+1)} = w^{(t)} + \frac{1}{K} \sum_{k=1}^K \delta^{(t-k)}
        Use the history weight delta to predict the next step.
        """ 
        pred_weight_dict = copy.deepcopy(weight_dict)

        for i, w_name in enumerate(weight_dict):
            weight = weight_dict[w_name]
            pred_weight_dict[w_name] = self._weight_buffer[i].output()

        return pred_weight_dict

    def update_buffer(self, weight_dict):
        for i, w_name in enumerate(weight_dict):
            weight = weight_dict[w_name]
            self._weight_buffer[i].push(weight)

class AdaMPredictor(Predictor):
    def __init__(self, config):
        self._delta_buffers = []
        self.device = config.device
        self.betas = config.betas
        self.step_size = config.adam_step_size
    
    def init_delta_buffers(self, weight_dict):
        for w_name, w_value in weight_dict.items():
            delta_buffer = AdaMBuffer(self.betas)
            delta_buffer.init_buffer_(w_value, device=self.device)
            self._delta_buffers.append(delta_buffer)

        self.buffer_len = len(self._delta_buffers)

    def predict(self, weight_dict):
        """
        Use the history weight delta to predict the next step.
        """
        pred_weight_dict = copy.deepcopy(weight_dict)
        if self._delta_buffers[0].step == 0:
            pass
        else:
            for i, w_name in enumerate(weight_dict):
                weight = weight_dict[w_name]
                pred_weight_dict[w_name] = weight - self.step_size*self._delta_buffers[i].delta()
            
        return pred_weight_dict

    def update_buffer(self, delta_weight_dict):
        for i, w_name in enumerate(delta_weight_dict):
            delta_weight = delta_weight_dict[w_name]
            self._delta_buffers[i].push(delta_weight)

class ExpAvgPredictor(Predictor):
    def __init__(self, config):
        self._delta_buffers = []
        self.device = config.device
        self.betas = config.betas[0]
    
    def init_delta_buffers(self, weight_dict):
        for w_name, w_value in weight_dict.items():
            delta_buffer = ExpAvgBuffer(self.betas)
            delta_buffer.init_buffer_(w_value, device=self.device)
            self._delta_buffers.append(delta_buffer)

        self.buffer_len = len(self._delta_buffers)

    def predict(self, weight_dict):
        """
        Use the history weight delta to predict the next step.
        """
        pred_weight_dict = copy.deepcopy(weight_dict)
        if self._delta_buffers[0].delta_ is None:
            pass
        else:
            for i, w_name in enumerate(weight_dict):
                weight = weight_dict[w_name]
                pred_weight_dict[w_name] = weight - self._delta_buffers[i].delta()
            
        return pred_weight_dict

    def update_buffer(self, delta_weight_dict):
        for i, w_name in enumerate(delta_weight_dict):
            delta_weight = delta_weight_dict[w_name]
            self._delta_buffers[i].push(delta_weight)


class DeltaStepPredictor(Predictor):
    def __init__(self, config):
        self._delta_buffers = []
        self.buffer_len = 0
        self.order = config.order
        self.device = config.device

    def init_delta_buffers(self, weight_dict):
        """Initialize delta buffer layer wise. 
        """
        for w_name, w_value in weight_dict.items():
            delta_buffer = DeltaBuffer(self.order)
            delta_buffer.init_buffer_(w_value, device=self.device)
            self._delta_buffers.append(delta_buffer)

        self.buffer_len = len(self._delta_buffers)

    def predict(self, weight_dict):
        """
        Use the history weight delta to predict the next step.
        """ 
        pred_weight_dict = copy.deepcopy(weight_dict)
        for i, w_name in enumerate(weight_dict):
            weight = weight_dict[w_name]
            pred_weight_dict[w_name] = weight - self._delta_buffers[i].delta()
        
        return pred_weight_dict

    def update_buffer(self, delta_weight_dict):
        for i, w_name in enumerate(delta_weight_dict):
            delta_weight = delta_weight_dict[w_name]
            self._delta_buffers[i].push(delta_weight)


