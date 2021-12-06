import numpy as np

# PyTorch Libraries
import torch

from fedlearning import Quantizer

"""
`Pos` means that quantizer will map the negative integers to positive integers. 
This step is for more efficient entropy coding. 
"""

class UniformPosQuantizer(Quantizer):
    def __init__(self, config):
        self.quant_level = config.quantization_level 
        self.quantbound = (config.quantization_level - 1)/2

        if self.quant_level % 2 == 0:   #  mid-riser quant, not mid-tread quant 
            self.mid_tread = False
        else:
            self.mid_tread = True

    def quantize(self, arr):
        """
        quantize a given arr array with uniform quantization.
        """
        max_val = torch.max(arr.abs())
        
        quant_step = max_val/self.quantbound

        if self.mid_tread:
            quantized_arr = torch.floor(arr/quant_step + 0.5)
        else:
            quantized_arr = torch.floor(arr/quant_step)
        
        quantized_arr = torch.where(quantized_arr>0, 2*quantized_arr-1, -2*quantized_arr)
        quantized_set = dict(norm=max_val, quantized_arr=quantized_arr)    

        return quantized_set
    
    def dequantize(self, quantized_set):
        """
        dequantize a given array which is uniformed quantized.
        """
        quant_arr = quantized_set["quantized_arr"]
        dequant_arr = torch.where(quant_arr%2==0, -0.5*quant_arr, 0.5*(quant_arr+1))
        dequant_arr = dequant_arr/self.quantbound
        dequant_arr = quantized_set["norm"] * dequant_arr

        return dequant_arr


class NormPosQuantizer(Quantizer):

    def __init__(self, config):
        
        self.quant_level = config.quantization_level 
        self.quantbound = (config.quantization_level - 1)/2
        if self.quant_level%2 == 0:
            self._cut_neg = True     
        else:
            self._cut_neg = False  #  mid-riser quant

        self.scale = 90

    def quantize(self, arr):
        norm = self.scale*torch.max(torch.abs(arr))
        abs_arr = arr.abs()

        level_float = abs_arr / norm * self.quantbound 
        lower_level = level_float.floor()
        rand_variable = torch.empty_like(arr).uniform_() 
        is_upper_level = rand_variable < (level_float - lower_level)
        new_level = (lower_level + is_upper_level)

        sign = arr.sign()
        quantized_arr = sign * torch.round(new_level).to(torch.int)

        if self._cut_neg:
            quantized_arr = torch.where(quantized_arr==-torch.ceil(torch.tensor(self.quantbound)),
                                        quantized_arr+1,
                                        quantized_arr)

        quantized_arr = torch.where(quantized_arr>0, 2*quantized_arr-1, -2*quantized_arr)

        quantized_set = dict(norm=norm, quantized_arr=quantized_arr)

        return quantized_set

    def dequantize(self, quantized_set):
        quant_arr = quantized_set["quantized_arr"]
        dequant_arr = torch.where(quant_arr%2==0, -0.5*quant_arr, 0.5*(quant_arr+1))
        dequant_arr = dequant_arr/self.quantbound
        dequant_arr = quantized_set["norm"] * dequant_arr

        return dequant_arr

class QsgdPosQuantizer(Quantizer):

    def __init__(self, config):
        
        self.quantlevel = config.quantization_level 
        self.quantbound = (config.quantization_level - 1)/2
        if self.quantlevel%2 == 0:
            self._cut_neg = True     
        else:
            self._cut_neg = False  #  mid-riser quant

    def quantize(self, arr):
        norm = arr.norm()
        abs_arr = arr.abs()

        level_float = abs_arr / norm * self.quantbound 
        lower_level = level_float.floor()
        rand_variable = torch.empty_like(arr).uniform_() 
        is_upper_level = rand_variable < (level_float - lower_level)
        new_level = (lower_level + is_upper_level)

        sign = arr.sign()
        quantized_arr = sign * torch.round(new_level).to(torch.int)

        if self._cut_neg:
            quantized_arr = torch.where(quantized_arr==-torch.ceil(torch.tensor(self.quantbound)),
                                        quantized_arr+1,
                                        quantized_arr)

        quantized_arr = torch.where(quantized_arr>0, 2*quantized_arr-1, -2*quantized_arr)

        quantized_set = dict(norm=norm, quantized_arr=quantized_arr)

        return quantized_set

    def dequantize(self, quantized_set):
        quant_arr = quantized_set["quantized_arr"]
        dequant_arr = torch.where(quant_arr%2==0, -0.5*quant_arr, 0.5*(quant_arr+1))
        dequant_arr = dequant_arr/self.quantbound
        dequant_arr = quantized_set["norm"] * dequant_arr

        return dequant_arr

class QsgdinfPosQuantizer(Quantizer):

    def __init__(self, config):
        
        self.quantlevel = config.quantization_level 
        self.quantbound = (config.quantization_level - 1)/2
        if self.quantlevel%2 == 0:
            self._cut_neg = True
        else:
            self._cut_neg = False

    def quantize(self, arr):
        norm = torch.max(arr.abs())
        abs_arr = arr.abs()

        level_float = abs_arr / norm * self.quantbound 
        lower_level = level_float.floor()
        rand_variable = torch.empty_like(arr).uniform_() 
        is_upper_level = rand_variable < (level_float - lower_level)
        new_level = (lower_level + is_upper_level)

        sign = arr.sign()
        quantized_arr = sign * torch.round(new_level).to(torch.int)

        if self._cut_neg:
            quantized_arr = torch.where(quantized_arr==-torch.ceil(torch.tensor(self.quantbound)),
                                        quantized_arr+1,
                                        quantized_arr)

        quantized_arr = torch.where(quantized_arr>0, 2*quantized_arr-1, -2*quantized_arr)
        quantized_set = dict(norm=norm, quantized_arr=quantized_arr)

        return quantized_set

    def dequantize(self, quantized_set):
        quant_arr = quantized_set["quantized_arr"]
        dequant_arr = torch.where(quant_arr%2==0, -0.5*quant_arr, 0.5*(quant_arr+1))
        dequant_arr = dequant_arr/self.quantbound
        dequant_arr = quantized_set["norm"] * dequant_arr

        return dequant_arr

# A quantizer that does nothing. (for vanilla FedAvg)
class PlainQuantizer(Quantizer):
    def __init__(self, config):
        pass

    def quantize(self, arr):
        """
        simply return the arr.
        """
        quantized_set = dict(quantized_arr=arr)
        return quantized_set
    
    def dequantize(self, quantized_set):
        """
        dequantize a given array which is uniformed quantized.
        """
        dequant_arr = quantized_set["quantized_arr"]

        return dequant_arr

class StcCompressor(Quantizer):

    def __init__(self, config):
        super().__init__()
        self.sparsity = config.sparsity

    def quantize(self, tensor, **kwargs):
        """
        Compress the input tensor with stc and simulate the saved data volume in bit.

        Args,
            tensor (torch.tensor): the input tensor.
        """
        k = np.ceil(tensor.numel()*self.sparsity).astype(int)
        top_k_element, top_k_index = torch.kthvalue(-tensor.abs().flatten(), k)
        tensor_masked = (tensor.abs() > -top_k_element) * tensor

        norm = (1/k) * tensor_masked.abs().sum()
        quantized_set = dict(norm=norm, quantized_arr=tensor_masked.sign())

        return quantized_set 

    def dequantize(self, quantized_set):
        """Decode the signs to float format """
        quantized_arr = quantized_set["quantized_arr"]
        norm = quantized_set["norm"]
        return norm*quantized_arr

