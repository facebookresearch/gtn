#!/usr/bin/env python3

import struct
import torch 
import math 

from gtn._torch import ctc_loss 

def get_data_ptr_as_bytes(tensor):
    return struct.pack("P", tensor.data_ptr())

class CTCLossFunction(torch.autograd.Function) : 
    @staticmethod 
    def forward(ctx, log_probs, targets, blank_idx=0) : 
        if log_probs.is_cuda:
            log_probs = log_probs.cpu()
        batch_size, N, T = list(log_probs.shape) 
        loss = torch.zeros(batch_size, dtype=torch.float)
        # TODO: do not compute grad if requires_grad is False
        input_grad = torch.zeros(log_probs.size(), dtype=torch.float)
        ctc_loss(
            get_data_ptr_as_bytes(log_probs),
            T,
            N,
            targets,
            [],
            blank_idx,
            get_data_ptr_as_bytes(loss),
            get_data_ptr_as_bytes(input_grad)
        )
        ctx.constant = input_grad
        return loss  
    
    @staticmethod 
    def backward(ctx, grad_output) :
        return ctx.constant *  grad_output.view(-1,1,1).expand(ctx.constant.size()), None, None
        

CTCLoss=CTCLossFunction.apply