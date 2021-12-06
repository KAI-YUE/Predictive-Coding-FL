from abc import ABC, abstractmethod

class Predictor(ABC):
    """Interface for predicting the weight tensor"""
    def __init__(self):
        pass
    
    @abstractmethod
    def predict(self, weight):
        """Predict the weight in the next step.
        """
        
class Quantizer(ABC):
    """Interface for quantizing and dequantizing a given tensor."""

    def __init__(self):
        pass

    @abstractmethod
    def quantize(self, seq):
        """Compresses a tensor with the given compression context, 
        and then returns it with the context needed to decompress it."""

    @abstractmethod
    def dequantize(self, quantized_set):
        """Decompress the tensor with the given decompression context."""

class EntropyCoder(ABC):
    """Interface for compressing and decompressing a given tensor."""

    def __init__(self):
        pass

    @abstractmethod
    def encode(self, seq):
        """Compresses a tensor with the given compression context, and then returns it with the context needed to decompress it."""

    @abstractmethod
    def decode(self, coded_set):
        """Decompress the tensor with the given decompression context."""

from fedlearning.predictor import *
from fedlearning.quantizer import *
from fedlearning.entropy_coder import *

predictor_registry = {
    "previous_frame":   PrevFramePredictor,
    "ou_ada":           OuAdaptivePredictor,
    "delta_step":       DeltaStepPredictor,
    "adaM":             AdaMPredictor
}

encoder_registry = {
    "plain":        PlainCoder,
    "entropy":      IdealCoder,
    "arithmetic":   ArithmeticCoder,
    "stc_entropy":  StcIdealCoder
}

quantizer_registry = {
    "plain":            PlainQuantizer,
    "qsgd_pos":         QsgdPosQuantizer,
    "qsgdinf":          QsgdinfPosQuantizer,
    "uniform_pos":      UniformPosQuantizer,
    "norm_pos":         NormPosQuantizer,
    "stc":              StcCompressor
}