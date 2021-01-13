import torch
from torchtext import data
import numpy as np
from torch.autograd import Variable


def nopeak_mask(size, opt):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)
    if opt.device == 'cuda':
        np_mask = np_mask.cuda()
    return np_mask

def create_masks(src, trg, opt, device):
    
    src_mask = (src != opt.src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size, opt)
        if opt.device == 'cuda':
            np_mask.cuda()
        trg_mask = trg_mask & np_mask
        
    else:
        trg_mask = None
    return src_mask, trg_mask
