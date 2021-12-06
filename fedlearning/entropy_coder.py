import numpy as np

# PyTorch libraries
import torch

from fedlearning import EntropyCoder
import fedlearning.constant as const

# A plain coder does nothing but calculate the number of bits 
# of quantizer output
class PlainCoder(EntropyCoder):
    def __init__(self, config):
        self.total_codewords = config.quantization_level
        if config.quantization_level == 2:
            self.ratio_ = 0                     # sign bits are overlapped with level bits
        elif config.quantization_level == 0: 
            self.ratio_ = const.FLOAT_BIT        # this is kept for FedAve testing only
        else:
            self.ratio_ = np.ceil(np.log2(self.total_codewords))

    def encode(self, seq):
        total_symbols = torch.prod(torch.tensor(seq.shape)).item()
        original_bits = total_symbols * const.FLOAT_BIT
        compressed_bits = self._compressed_bits(total_symbols)

        coded_set = dict(code=seq,
                    compressed_bits=compressed_bits,
                    original_bits=original_bits)

        return coded_set

    def decode(self, coded_set):
        """
        Plain decoder directly returns the code. 
        """
        return coded_set["code"]

    def _compressed_bits(self, total_symbols):
        magnitude_bits = 1 * const.FLOAT_BIT
        if self.ratio_ == const.FLOAT_BIT:
            return total_symbols*self.ratio_

        return total_symbols*self.ratio_ + magnitude_bits

        

# An ideal coder uses the histogram as an estimation of entropy
# and assume the entropy lower bound can be achieved. Nothing is 
# actually done but the number of estimated bits will be returned 
class IdealCoder (EntropyCoder):
    def __init__(self, config):
        self.total_codewords = config.quantization_level

    def encode(self, seq):
        """
        Simulate an ideal entropy coding to a quantized tensor 
        without actually coding the tensor array. 
        """
        histogram = torch.histc(seq, bins=self.total_codewords, min=0, max=self.total_codewords-1) 
        total_symbols = torch.sum(histogram).item()
        entropy = self._entropy((histogram.to(torch.float)/total_symbols).detach().cpu().numpy())
        
        original_bits = total_symbols * const.FLOAT_BIT
        compressed_bits = self._compressed_bits(total_symbols, entropy)

        coded_set = dict(code=seq,
                    compressed_bits=compressed_bits,
                    original_bits=original_bits)

        return coded_set
                    
    def decode(self, coded_set):
        """
        Simulate an ideal entropy decoding without actually decoding the tensor array. 
        """
        return coded_set["code"]

    def _compressed_bits(self, total_symbols, ratio):
        magnitude_bits = 1 * const.FLOAT_BIT

        return total_symbols*ratio + magnitude_bits

    @staticmethod
    def _entropy(histogram):
        entropy = 0

        for i, prob in enumerate(histogram):
            if prob == 0:
                continue
            entropy += -prob * np.log2(prob)

        return entropy

# ideal entropy coder for stc compressor
class StcIdealCoder (EntropyCoder):
    def __init__(self, config):
        self.total_codewords = 3    # ternary quantization

    def encode(self, seq):
        """
        Simulate an ideal entropy coding to a quantized tensor 
        without actually coding the tensor array. 
        """
        histogram = torch.histc(seq, bins=self.total_codewords, min=-1, max=1) 
        total_symbols = torch.sum(histogram).item()
        entropy = self._entropy((histogram.to(torch.float)/total_symbols).detach().cpu().numpy())
        
        original_bits = total_symbols * const.FLOAT_BIT
        compressed_bits = self._compressed_bits(total_symbols, entropy)

        coded_set = dict(code=seq,
                    compressed_bits=compressed_bits,
                    original_bits=original_bits)

        return coded_set
                    
    def decode(self, coded_set):
        """
        Simulate an ideal entropy decoding without actually decoding the tensor array. 
        """
        return coded_set["code"]

    def _compressed_bits(self, total_symbols, ratio):
        magnitude_bits = 1 * const.FLOAT_BIT
        
        return total_symbols*ratio+ magnitude_bits

    @staticmethod
    def _entropy(histogram):
        entropy = 0

        for i, prob in enumerate(histogram):
            if prob == 0:
                continue
            entropy += -prob * np.log2(prob)

        return entropy

# Arithmetic Coding
# References:
#         [1] Sayood, K., Introduction to Data Compression, 
#         Morgan Kaufmann, 2017, Chapter 4, Section 4.4.3.

class ArithmeticCoder(EntropyCoder):
    def __init__(self, config):
        self.total_codewords = config.quantization_level
        self.cumulative_hist = None

    def encode(self, seq):
        histogram = torch.histc(seq, bins=self.total_codewords, min=0, max=self.total_codewords-1) 
        # Laplacian smoothing
        histogram += 1
        
        cumulative_hist = self.cumulative_sum_(histogram)
        total_symbols = cumulative_hist[-1]

        N = np.ceil(np.log2(total_symbols)) + 2
        N = int(N)
        
        # Initialize the lower and upper bounds
        low = 0
        up = 2**N-1
        E3_count = 0
        code = []

        # Initialize the bit mask
        N_bitmask = 2**N - 1 
        N_minusone_bitmask = 2**(N-1)

        seq_flatten = seq.flatten()
        for i, symbol in enumerate(seq_flatten):
            symbol = int(symbol.item())
            # update the lower bound & upper bound
            range_ = up - low + 1
            low_new = int(low + np.floor(range_*cumulative_hist[symbol]/total_symbols))
            up = int(low + np.floor(range_*cumulative_hist[symbol+1]/total_symbols)-1)
            low = low_new

            # Check for E1, E2 or E3 conditions and keep looping as long as they occur.
            # For the details of E1,2,3, please refer to the reference[1].
            E1_2 = (low>>(N-1) == up>>(N-1))
            E3 = (low>>(N-2) & 1 == 1) and (up>>(N-2) & 1 == 0)

            while(E1_2 or E3):
                if E1_2:
                    MSB = low>>(N-1)
                    code.append(MSB)
                    low = low<<1
                    up = (up<<1) ^ 1      

                    # Check if E3_count is non-zero and transmit appropriate bits
                    if E3_count > 0:
                        for j in range(E3_count, 0, -1):
                            # Have to transmit complement of MSB, E3_count times.
                            code.append(1-MSB)
                        E3_count = 0
                        
                    low = low & N_bitmask
                    up = up & N_bitmask
            
                elif E3:
                    low = low<<1
                    low = low & N_bitmask 
                    low = low ^ N_minusone_bitmask
                    up  = (up<<1) ^ 1
                    up = up & N_bitmask
                    up  = up ^ N_minusone_bitmask
                    E3_count += 1

                # update E1,2,3
                E1_2 = (low>>(N-1) == up>>(N-1))
                E3 = (low>>(N-2) & 1 == 1) and (up>>(N-2) & 1 == 0)
        
        # terminate coding
        if E3_count==0:
            # Just transmit the final value of the lower bound    
            for i in range(N-1,-1,-1):
                code.append((low>>i)&1)   
        else:
            # Transmit the MSB of bin_low. 
            MSB = low>>(N-1)
            code.append(MSB)

            # Then transmit complement of b (MSB of bin_low), E3_count times. 
            for i in range(E3_count):
                code.append(1-MSB)

            # Then transmit the remaining bits of bin_low
            for i in range(N-2,-1,-1):
                code.append((low>>i)&1)
        
        # cancel out the influence of Laplacian smoothing
        total_symbols -= self.total_codewords
        original_bits = total_symbols*const.FLOAT_BIT
        compressed_bits = self._compressed_bits(total_symbols, len(code))

        coded_set = dict(code=code,
                    compressed_bits=compressed_bits,
                    original_bits=original_bits,
                    cumulative_hist = cumulative_hist.copy())

        return coded_set
    
    def decode(self, coded_set):
        code = coded_set["code"]
        total_symbols = coded_set["original_bits"]/const.FLOAT_BIT
        total_symbols = int(total_symbols)

        cumulative_hist = coded_set["cumulative_hist"]

        # Compute the Word Length (N) required.
        total_count = total_symbols + self.total_codewords
        N = int(np.ceil(np.log2(total_count)) + 2)
        
        # Initialize the lower and upper bounds.
        low= 0
        up = 2**N-1
        
        # Read the first N number of bits into a temporary tag bin_tag
        bin_tag = code[:N]
        tag = bi2de(bin_tag)
        
        # Initialize DSEQ
        decoded_seq = torch.zeros(total_symbols)
        decoded_seq_index = 0
        k = N - 1
        
        # Initialize the bit mask
        N_bitmask = 2**N - 1 
        N_minusone_bitmask = 2**(N-1)
        
        # This loop runs until all the symbols are decoded into DSEQ
        while (decoded_seq_index < total_symbols):
            range_ = up - low + 1

            if range_ <= 0:
                np.savetxt("ori_seq.txt",coded_set["original_seq"].cpu().numpy().flatten())
                np.savetxt("hist.txt", coded_set["cumulative_hist"])

            # Compute tag_new and decode a symbol based on tag_new
            tag_new = np.floor(((tag-low+1)*total_count-1)/range_)
            ptr = pick(cumulative_hist, tag_new)
            
            if ptr == None:
                np.savetxt("ori_seq.txt",coded_set["original_seq"].cpu().numpy().flatten())
                np.savetxt("hist.txt", coded_set["cumulative_hist"])

            decoded_seq[decoded_seq_index] = ptr
            decoded_seq_index += 1
            
            # Compute the new lower bound
            low_new = low + np.floor(range_*cumulative_hist[ptr]/total_count)
            
            # Compute the new upper bound
            up = int(low + np.floor(range_*cumulative_hist[ptr+1]/total_count) -1)
            low = int(low_new)
            
            # Check for E1, E2 or E3 conditions and keep looping as long as they occur.
            # For the details of E1,2,3, please refer to the reference[1].
            E1_2 = (low>>(N-1) == up>>(N-1))
            E3 = (low>>(N-2) & 1 == 1) and (up>>(N-2) & 1 == 0)
            while (E1_2 or E3):
                # Break out if we have finished working with all the bits in CODE
                if (k==len(code)-1):
                    break
                k += 1    
                if E1_2:
                    # Left shifts and update
                    low = low<<1
                    up  = (up<<1) ^ 1
                    # Left shift and read in code
                    tag = (tag<<1) ^ code[k]
        
                    # Reduce to N for next loop
                    low = low & N_bitmask
                    up  = up & N_bitmask
                    tag = tag & N_bitmask
                
                # Else if it is an E3 condition        
                elif E3: 
                    low = low<<1
                    low = low & N_bitmask
                    low= low ^ N_minusone_bitmask
                    
                    up  = (up<<1) ^ 1
                    up  = up & N_bitmask
                    up  = up ^ N_minusone_bitmask
                    
                    tag = (tag<<1) ^ code[k]
                    tag = tag & N_bitmask
                    tag = tag ^ N_minusone_bitmask
                    
                E1_2 = (low>>(N-1) == up>>(N-1))
                E3 = (low>>(N-2) & 1 == 1) and (up>>(N-2) & 1 == 0)
            
        return decoded_seq

    @staticmethod
    def cumulative_sum_(histogram):
        cumulative_hist = np.zeros(histogram.shape[0]+1)
        for i in range(1, cumulative_hist.shape[0]):
            cumulative_hist[i] = cumulative_hist[i-1] + histogram[i-1]

        return cumulative_hist

    def _compressed_bits(self, total_symbols, bitstream_len):
        magnitude_bits = 1 * const.FLOAT_BIT
        prob_tab_bits = self.total_codewords * const.FLOAT_BIT
        
        return bitstream_len + magnitude_bits + prob_tab_bits

def bi2de(bin_code):
    decimal_num = 0
    for bin_symbol in bin_code: 
        decimal_num = decimal_num^bin_symbol
        decimal_num = decimal_num << 1
    
    return decimal_num>>1

def pick(cumulative_hist, value):
    # find where value is positioned

    # Check for this case and quickly exit
    if value >= cumulative_hist[-1]:
        ptr = len(cumulative_hist)-2
        return ptr
    
    for ptr, cum_count in enumerate(cumulative_hist):
        if (cum_count > value and cumulative_hist[ptr-1] <=value):
            return ptr-1

class ContextModel(object):
    def __init__(self, config):
        """Construct a global context model for arithmetic coding.
        """
        self.context_len = config.context_len
        self.context_ptr = 0
        self.context_buffer_empty = True
        self.context_buffer_full = False
        self.cumulative_counts = None
        self.counts_pool = np.zeros((self.context_len, config.quantization_level))
    
    def _update_context(self, histogram):
        self.context_buffer_empty = False
        if type(histogram) == torch.Tensor:
            histogram = histogram.detach().cpu().numpy()

        self.counts_pool[self.context_ptr] = histogram
        self.context_ptr += 1
        if (self.context_ptr == self.context_len):
            self.context_ptr = 0
            if (not self.context_buffer_full):
                self.context_buffer_full = True

        if self.context_buffer_full:
            counts = np.sum(self.counts_pool, axis=0)
        else:
            counts = np.sum(self.counts_pool[:self.context_ptr], axis=0)

        self.cumulative_counts = cumulative_sum_(counts)