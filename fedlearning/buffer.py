import copy
import math

# PyTorch Libraries
import torch
import torch.nn as nn
import torch.optim as optim

# My Libraries
from config.loadconfig import load_config
from deeplearning import nn_registry
from deeplearning.networks import init_weights

# DeltaBuffer to keep history global weight 
class DeltaBuffer(object):
    def __init__(self, delta_k):
        self._buffer = []
        self.buffer_full_ = False
        self._buffer_len = delta_k 
        self.top_ = -1

    def init_buffer_(self, weight_template, device="cuda"):
        for k in range(self._buffer_len):
            self._buffer.append(torch.zeros_like(weight_template))

    def push(self, delta_weight):
        self.top_ += 1
        if self.top_ ==  self._buffer_len:
            self.top_ = 0
            self.buffer_full_ = True

        self._buffer[self.top_] = delta_weight.clone()

    def delta(self):
        if self.buffer_full_:
            delta_ = self._buffer[0].clone()
            for i in range(1, self._buffer_len):
                delta_ += self._buffer[i]
            delta_ *= 1/self._buffer_len 
        elif self.top_ >= 0:
            # delta_ = self._buffer[0].clone()
            # for i in range(1, self.top_+1):
            #     delta_ += self._buffer[i]
            # delta_ *= 1/(self.top_ + 1)
            delta_ = torch.zeros_like(self._buffer[0])
        else:
            delta_ = self._buffer[0].clone()

        return delta_

# WeightBuffer to keep history global weight 
class WeightMomentBuffer(object):
    def __init__(self, order):
        self._buffer = []
        self.buffer_full_ = False
        self._buffer_len = order 
        self.top_ = -1
        self.mean = None

    def init_buffer_(self, weight_template, device="cuda"):
        for k in range(self._buffer_len):
            self._buffer.append(weight_template.clone())

    def push(self, weight):
        self.top_ += 1
        if self.top_ ==  self._buffer_len:
            self.top_ = 0
            self.buffer_full_ = True

        self._buffer[self.top_] = weight.clone()

    def first_moment(self):
        if self.buffer_full_:
            weight_ = self._buffer[0].clone()
            for i in range(1, self._buffer_len):
                weight_ += self._buffer[i]
            weight_ *= 1/self._buffer_len
            
        elif self.top_ >= 0:
            weight_ = self._buffer[0].clone()
            for i in range(1, self.top_+1):
                weight_ += self._buffer[i]
            weight_ *= 1/(self.top_ + 1)
        else:
            weight_ = self._buffer[0].clone()
        
        self.mean = weight_.clone()
        return weight_

    def variance(self):
        assert (self.mean is not None), ""
        if self.buffer_full_:
            weight_sq_ = (self._buffer[0] - self.mean) **2
            for i in range(1, self._buffer_len):
                weight_sq_ += (self._buffer[i] - self.mean)**2
            weight_sq_ *= 1/(self._buffer_len-1) 
        elif self.top_ >= 2:
            weight_sq_ = (self._buffer[0] - self.mean)**2
            for i in range(1, self.top_+1):
                weight_sq_ += (self._buffer[i] - self.mean)**2
            weight_sq_ *= 1/(self.top_ + 1)
        else:
            weight_sq_ = (self._buffer[0] - self.mean)**2

        return weight_sq_

class OuArBuffer(object):
    def __init__(self):
        self._buffer = []
        self.buffer_full_ = False
        self._buffer_len = 2        # for the ou process, we set buffer length to 2
        self.top_ = 0
        self.shape = None

    def init_buffer_(self, weight_template, device="cuda"):
        for k in range(self._buffer_len):
            self._buffer.append(weight_template.clone().flatten().view(-1,1))
        
        self.shape = weight_template.shape

    def push(self, weight):
        if self.top_ == -1:
            self.top = 0
        elif self.top_ == 0:
            self.buffer_full_ = True
            self.top = 1
        elif self.top == 1:
            self._buffer[0] = self._buffer[1].clone()
        
        self._buffer[self.top_] = weight.clone().flatten().view(-1,1)
            
    def output(self):
        if self.buffer_full_:
            prev_weight = self._buffer[0]
            beta = torch.inverse(prev_weight.T @ prev_weight) @ prev_weight.T @ self._buffer[1]
            out = beta*self._buffer[1].clone()
        else:
            out = self._buffer[1].clone()

        out = out.view(self.shape)

        return out


# ArDeltaBuffer predicts delta with dynamic AR coefficients
class ArDeltaBuffer(object):
    def __init__(self, order):
        self._buffer = []
        self.buffer_full_ = False
        self.order = order
        self._buffer_len = order + 1
        self.top_ = -1
        self.shape = None

    def init_buffer_(self, weight_template, device="cuda"):
        for k in range(self._buffer_len):
            self._buffer.append(torch.zeros_like(weight_template))
        
        self.shape = weight_template.shape

    def push(self, delta_weight):
        self.top_ += 1
        if self.top_ ==  self._buffer_len:
            self.top_ = 0
            self.buffer_full_ = True
        
        self._buffer[self.top_] = delta_weight.clone().flatten().view(-1,1)
            
    def delta(self):
        if self.buffer_full_:
            buffer_copy = copy.deepcopy(self._buffer)
            top_delta = buffer_copy.pop(self.top_)
            aug_delta = torch.cat(buffer_copy, dim=1)
            betas = torch.inverse(aug_delta.T @ aug_delta) @ aug_delta.T @ top_delta

            delta_ = betas[self.top_-1] * top_delta
            for i in range(1, self.order):
                delta_ += self._buffer[self.top_-i] * betas[self.top_-i-1]
        else:
            delta_ = self._buffer[-1].clone()

        delta_ = delta_.view(self.shape)

        return delta_

class AdaMBuffer(object):
    def __init__(self, betas):
        self.exp_avg = None
        self.exp_avg_sq = None
        self.betas = betas
        self.step = 0
        self.eps = 1.e-4
        self.delta_ = None

    def init_buffer_(self, weight_template, device="cuda"):
        self.exp_avg = torch.zeros_like(weight_template)
        self.exp_avg_sq = torch.zeros_like(weight_template)

    def push(self, delta_weight):
        self.step += 1
        
        self.exp_avg = self.betas[0]*self.exp_avg + (1-self.betas[0])*delta_weight
        self.exp_avg_sq = self.betas[1]*self.exp_avg_sq + (1-self.betas[1])*delta_weight**2

        # bias_correction1 = 1 - self.betas[0] ** self.step
        # bias_correction2 = 1 - self.betas[1] ** self.step

        # denom = ((self.exp_avg_sq).sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
        # self.delta_ = (1./bias_correction1) * (self.exp_avg/denom)

        denom = (self.exp_avg_sq).sqrt().add_(self.eps)
        self.delta_ = (self.exp_avg/denom)

    def delta(self):
        return self.delta_

class ExpAvgBuffer(object):
    def __init__(self, beta):
        self.exp_avg = None
        self.beta = beta
        self.delta_ = None

    def init_buffer_(self, weight_template, device="cuda"):
        self.exp_avg = torch.zeros_like(weight_template)

    def push(self, delta_weight):
        self.exp_avg = self.beta*self.exp_avg + (1-self.beta)*delta_weight
        self.delta_ = self.exp_avg

    def delta(self):
        return self.delta_

# Adaptive OU Buffer
class OuAdaptiveBuffer(object):
    def __init__(self, step_size, scaler):
        self._buffer = []
        self.buffer_full_ = False
        self.order = 1
        self._buffer_len = 2
        self.top_ = 0

        # learnable coefficients
        self.coeffs = []
        self.bias = []
        self.scaler = scaler
        self.step_size = step_size

    def init_buffer_(self, weight_template, device="cuda"):
        template = weight_template.flatten()
        for k in range(self._buffer_len):
            self._buffer.append(weight_template.detach().clone())
        
        self.shape = weight_template.shape
        
        # initialize learnable  coefficients.params and the corresponding optimizer
        self.coeffs.append(torch.ones_like(weight_template, requires_grad=True))  # coefficients
        self.bias.append(torch.randn_like(weight_template, requires_grad=True))  # bias
        self.coeffs[0].data *= self.scaler
        self.bias[0].data *= 0.
        
        self.optimizer = optim.AdamW([
            {"params": self.coeffs_params(), "lr": self.step_size[0]},
            {"params": self.bias_params(), "lr": self.step_size[1]}],
            weight_decay=1.e-3)

        # self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        # self.optimizer = optim.SGD(self.learnable_params(), lr=self.step_size)

    def coeffs_params(self):
        for param in self.coeffs:
            yield param
    
    def bias_params(self):
        for param in self.bias:
            yield param

    def push(self, weight):
        if self.top_ == 0:
            self.buffer_full_ = True
            self.top_ = 1
        elif self.top_ == 1:
            self._buffer[0] = self._buffer[1].clone()
        
        self._buffer[self.top_] = weight.detach().clone()
            
    def output(self):
        out = self.bias[0] + self.coeffs[0] * self._buffer[1]
        self.predict_register = out.clone()
        return out.data

    def optimizer_step(self):
        if not self.buffer_full_:
            pass
        else:
            self.optimizer.zero_grad()
            mse = torch.norm(self.predict_register - self._buffer[1])**2
            # mse = torch.norm(self.predict_register - torch.zeros_like(self.predict_register))**2
            # print("-- mse {:.3e} --".format(mse))
            mse.backward()
            # loss.backward()

            self.optimizer.step()
            # self.lr_scheduler.step()

# AdaptiveDeltaBuffer predicts delta with dynamic learnable coefficients
class AdaptiveDeltaBuffer(object):
    def __init__(self, order, step_size):
        self._buffer = []
        self.buffer_full_ = False
        self.order = order
        self._buffer_len = order + 1
        self.top_ = -1

        # learnable coefficients
        self.params = []
        self.step_size = step_size

    def init_buffer_(self, weight_template, device="cuda"):
        template = weight_template.flatten()
        for k in range(self._buffer_len):
            self._buffer.append(torch.zeros_like(weight_template.flatten()))
        
        self.shape = weight_template.shape
        self.numel = template.numel()
        
        # initialize learnable  coefficients.params and the corresponding optimizer
        self.params.append(torch.ones((self.order, self.numel), requires_grad=True, device=device))  # coefficients
        self.params.append(torch.zeros(self.numel, requires_grad=True, device=device))  # bias
        self.params[0].data *= 0.1
        self.optimizer = optim.Adam(self.learnable_params(), lr=self.step_size)
        # self.optimizer = optim.SGD(self.learnable_params(), lr=self.step_size)

    def learnable_params(self):
        for beta in self.params:
            yield beta

    def push(self, delta_weight):
        self.top_ += 1
        if self.top_ ==  self._buffer_len:
            self.top_ = 0
            self.buffer_full_ = True
        
        self._buffer[self.top_] = delta_weight.data.flatten()
            
    def delta(self):
        if self.buffer_full_:
            # delta_ = self.params[1]
            delta_ = torch.zeros_like(self.params[1])
            for i in range(0, self.order):
                delta_ = delta_ + self.params[0][i] * self._buffer[self.top_-i] 
        elif self.top_ == self._buffer_len - 1:
            # delta_ = self.params[1]
            delta_ = torch.zeros_like(self.params[1])
            for i in range(0, self.order):
                delta_ = delta_ + self.params[0][i] * self._buffer[self.top_-i]
        else:
            delta_ = self._buffer[-1]

        self._delta_register = delta_.clone()
        delta_ = delta_.view(self.shape)

        return delta_

    def optimizer_step(self):
        if not self.buffer_full_:
            pass
        else:
            self.optimizer.zero_grad()
            # mse = torch.norm(self._delta_register - self._buffer[self.top_])**2
            # print("-- mse {:.3e} --".format(mse))
            # mse.backward()
            cosine_sim = (self._delta_register @ self._buffer[self.top_])/(torch.norm(self._delta_register)*torch.norm(self._buffer[self.top_]))
            loss = 1 - cosine_sim
            print("--- {:.3f} ---".format(loss))
            loss.backward()

            self.optimizer.step()

class WeightBuffer(object):
    def __init__(self, weight_dict, mode="copy"):
        self._weight_dict = copy.deepcopy(weight_dict)
        if mode == "zeros":
            for w_name, w_value in self._weight_dict.items():
                self._weight_dict[w_name].data = torch.zeros_like(w_value)
        
    def __add__(self, weight_buffer):
        weight_dict = copy.deepcopy(self._weight_dict)
        for w_name, w_value in weight_dict.items():
            weight_dict[w_name].data = self._weight_dict[w_name].data + weight_buffer._weight_dict[w_name].data

        return WeightBuffer(weight_dict)

    def __sub__(self, weight_buffer):
        weight_dict = copy.deepcopy(self._weight_dict)
        for w_name, w_value in weight_dict.items():
            weight_dict[w_name].data = self._weight_dict[w_name].data - weight_buffer._weight_dict[w_name].data

        return WeightBuffer(weight_dict)

    def __mul__(self,rhs):
        weight_dict = copy.deepcopy(self._weight_dict)
        for w_name, w_value in weight_dict.items():
            weight_dict[w_name].data = rhs*self._weight_dict[w_name].data

        return WeightBuffer(weight_dict)

    def push(self, weight_dict):
        self._weight_dict = copy.deepcopy(weight_dict)

    def state_dict(self):
        return self._weight_dict