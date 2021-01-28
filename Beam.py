import torch
from Batch import nopeak_mask
import math


def nonzero(t, v):
    pass

def init_vars(src, model, src_vocab, trg_vocab, opt):
    
    init_tok = trg_vocab.bos_idx
    src_mask = (src != src_vocab.pad_idx).unsqueeze(-2).to(opt.device)
    e_output = model.encoder(src, src_mask)
    
    outputs = torch.LongTensor([[init_tok]]).to(opt.device)
    
    trg_mask = nopeak_mask(1, opt).to(opt.device)
    
    out = model.out(model.decoder(outputs, e_output, src_mask, trg_mask))
    
    probs, ix = out[:, -1].data.topk(opt.beam_size)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    
    outputs = torch.zeros(opt.beam_size, opt.max_trg_len).long().to(opt.device)
    outputs = outputs.to(opt.device)
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]
    
    e_outputs = torch.zeros(opt.beam_size, e_output.size(-2),e_output.size(-1))
    e_outputs = e_outputs.to(opt.device)
    e_outputs[:, :] = e_output[0]
    
    return outputs, e_outputs, log_scores

def k_best_outputs(outputs, out, log_scores, i, k):
    
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
    k_probs, k_ix = log_probs.view(-1).topk(k)
    
    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)
    
    return outputs, log_scores

def beam_search(src, model, src_vocab, trg_vocab, opt):
    
    model.eval()
    outputs, e_outputs, log_scores = init_vars(src, model, src_vocab, trg_vocab, opt)
    eos_tok = trg_vocab.eos_idx
    src_mask = (src != src_vocab.pad_idx).unsqueeze(-2)
    ind = None
    for i in range(2, opt.max_trg_len):
    
        trg_mask = nopeak_mask(i, opt)

        out = model.out(model.decoder(outputs[:,:i], e_outputs, src_mask, trg_mask))
    
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, opt.beam_size)
        ones = (outputs == eos_tok).nonzero(as_tuple=True) # Occurrences of end symbols for all input sentences.
        x, y = ones
        ones = list(zip(x.detach().cpu().numpy(), y.detach().cpu().numpy()))
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).to(opt.device)

        for vec in ones:
            i = vec[0]
            if sentence_lengths[i] == 0: # First end symbol has not been found yet
                sentence_lengths[i] = vec[1] # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == opt.beam_size:
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break

    pad_token = trg_vocab.pad_idx

    if ind is None:
        length = (outputs[0] == eos_tok).nonzero(as_tuple=True)[0]
        outputs = outputs.detach().cpu().numpy()
        try:
            return ' '.join([trg_vocab.itos[tok] for tok in outputs[0][1:length]])
        except:
            return ' '.join([trg_vocab.itos[tok] for tok in outputs[0][1:]])
    
    else:
        length = (outputs[ind] == eos_tok).nonzero(as_tuple=True)[0]
        outputs = outputs.detach().cpu().numpy()
        try:
            return ' '.join([trg_vocab.itos[tok] for tok in outputs[ind][1:length] if tok != pad_token and tok != eos_tok])
        except:
            return ' '.join([trg_vocab.itos[tok] for tok in outputs[ind][1:] if tok != pad_token and tok != eos_tok])
