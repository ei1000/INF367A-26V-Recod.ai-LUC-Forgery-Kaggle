import torch
import torch.nn as nn
from datatypes import DLFDecoderInput
# TODO: Decoder which takes multi-scale errors and offset masks.
# Outputs sigmoid for each pixel


class DLFDecoder(nn.Module):
    def __init__(self, input: DLFDecoderInput):
        super().__init__()
        self.input = input