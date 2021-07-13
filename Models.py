import torch
from transformers import *
import torch.nn as nn
from Layers import EncoderLayer, DecoderLayer
from Embed import Embedder, PositionalEncoder
import phoBert_embed
from Sublayers import Norm
import torch.nn.functional as F
import copy


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(Encoder, self).__init__(config)
        self.roberta = RobertaModel(config)

        self.init_weights()

    def forward(self, src, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):
        outputs = self.roberta(src,
                               attention_mask=attention_mask,
                               # token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)  # [batch_size, max_seq_len, 768]
        # outputs[0]: last hidden layer
        # outputs[1]: unknown :(
        # outputs[2]: all hidden layers
        # get [CLS] of 4 last hidden layer
        # [:,0,:] = [batch_size, time_step_0, hidden_size]
        cls_output = outputs[0]
        return cls_output


class Decoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        # self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask, device):
        # x = self.embed(trg)
        x = phoBert_embed.build_sample_tensor(trg, device=device)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, trg_vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder.from_pretrained('roberta-base', output_hidden_states=True)
        self.decoder = Decoder(d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab_size)

    def forward(self, src, trg, src_mask, trg_mask, device):
        e_outputs = self.encoder(src, src_mask)  # [batch_size, max_len+1, d_model]
        # print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask.unsqueeze(-2), trg_mask, device)
        output = self.out(d_output)
        output = F.log_softmax(output, dim=-1)
        return output


def init_model(opt, trg_vocab_size, checkpoint=None):
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(trg_vocab_size, opt.d_model, opt.n_layers, opt.heads, opt.dropout).to(opt.device)

    if checkpoint is not None:
        print('load weight ...')
        model.load_state_dict(checkpoint['model'])
        if opt.device == 'cuda':
            torch.cuda.empty_cache()
    else:
        for p in model.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    return model
