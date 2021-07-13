import argparse
import time
import torch
from Models import init_model
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import joblib as pickle
import utils
from vocabulary import Vocabulary
from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def cal_performance(pred, trg_output, trg_pad_idx):
    ''' Apply label smoothing if needed '''

    pred = pred.max(1)[1]
    trg_output = trg_output.contiguous().view(-1)
    non_pad_mask = trg_output.ne(trg_pad_idx)
    n_correct = pred.eq(trg_output).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return n_correct, n_word


def train_epoch(model, optimizer_encoder, scheduler_encoder, optimizer_decoder, train_data, opt, epoch, start_time):
    model.train()

    total_loss, n_word_total, n_word_correct = 0, 0, 0
    print("   %dm: epoch %d [%s]  %d%%  loss = %s" % \
          ((time.time() - start_time) // 60, epoch + 1, "".join(' ' * 20), 0, '...'), end='\r')

    for i, batch in enumerate(train_data):

        src = batch[0].to(opt.device)
        trg = batch[1].to(opt.device)

        trg_input = trg[:, :-1]
        trg_output = trg[:, 1:].contiguous().view(-1)

        src_mask, trg_mask = create_masks(src, trg_input, opt)
        preds = model(src, trg_input, src_mask, trg_mask)
        preds = preds.view(-1, preds.size(-1))

        n_correct, n_word = cal_performance(preds, trg_output, opt.trg_pad)

        optimizer_decoder.zero_grad()
        optimizer_encoder.zero_grad()

        loss = F.cross_entropy(preds, trg_output, ignore_index=opt.trg_pad)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)

        optimizer_decoder.step()
        if opt.SGDR == True:
            opt.sched.step()

        # update parameters
        optimizer_encoder.step()
        # Update the learning rate.
        scheduler_encoder.step()

        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

        if (i + 1) % opt.print_every == 0:
            p = int(100 * (i + 1) / len(train_data))
            avg_loss = total_loss / (i + 1)
            print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" % \
                  ((time.time() - start_time) // 60, epoch + 1, "".join('#' * (p // 5)), "".join(' ' * (20 - (p // 5))),
                   p,
                   avg_loss))

    return total_loss, n_word_total, n_word_correct


def eval_epoch(model, valid_data, opt):
    model.eval()

    total_loss, n_word_total, n_word_correct = 0, 0, 0

    with torch.no_grad():
        for i, batch in enumerate(valid_data):
            src = batch[0].to(opt.device)
            trg = batch[1].to(opt.device)

            trg_input = trg[:, :-1]
            trg_output = trg[:, 1:].contiguous().view(-1)

            src_mask, trg_mask = create_masks(src, trg_input, opt)
            preds = model(src, trg_input, src_mask, trg_mask)
            preds = preds.view(-1, preds.size(-1))

            n_correct, n_word = cal_performance(preds, trg_output, opt.trg_pad)

            loss = F.cross_entropy(preds, trg_output, ignore_index=opt.trg_pad)

            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    return total_loss, n_word_total, n_word_correct


def train(model, optimizer_encoder, scheduler_encoder, optimizer_decoder, train_data, valid_data, opt):
    print("training model...")
    start = time.time()

    utils.mkdir('models')

    n_patience, pre_valid_loss, best_valid_acc = 0, 0, 0

    for epoch in range(opt.epochs):

        total_train_loss, n_word_total, n_word_correct = train_epoch(model, optimizer_encoder, scheduler_encoder,
                                                                     optimizer_decoder, train_data, opt, epoch, start)

        train_accuracy = n_word_correct / n_word_total
        avg_train_loss = total_train_loss / len(train_data)

        total_valid_loss, n_word_total, n_word_correct = eval_epoch(model, valid_data, opt)
        valid_accuracy = n_word_correct / n_word_total
        avg_valid_loss = total_valid_loss / len(valid_data)

        if valid_accuracy > best_valid_acc:
            best_valid_acc = valid_accuracy
            checkpoint = {'epoch': epoch, 'settings': opt, 'model': model.state_dict(),
                          'best_model': model.state_dict()}
        else:
            checkpoint = {'epoch': epoch, 'settings': opt, 'model': model.state_dict()}
        torch.save(checkpoint, 'models/checkpoint.chkpt')

        if opt.patience > 0 and avg_valid_loss >= pre_valid_loss:
            n_patience += 1
        pre_valid_loss = avg_valid_loss

        print("   %dm: epoch %d [%s%s]  %d%%\ntrain_loss = %.3f  train_acc=%.3f\nvalid_loss = %.3f  valid_acc = %.3f" % \
              ((time.time() - start) // 60, epoch + 1, "".join('#' * (100 // 5)), "".join(' ' * (20 - (100 // 5))), 100,
               avg_train_loss, train_accuracy, avg_valid_loss, valid_accuracy))

        if opt.patience > 0 and n_patience >= opt.patience:
            print('early stopping...')
            break


def test(model, test_data, opt):
    print('testing...')
    total_loss, n_word_total, n_word_correct = eval_epoch(model, test_data, opt)
    acc = n_word_correct / n_word_total
    avg_loss = total_loss / len(test_data)
    print('test_loss = %.3f  test_acc=%.3f' % (avg_loss, acc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-d_model', type=int, default=768)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batch_size', type=int, default=512)
    parser.add_argument('-print_every', type=int, default=10)
    parser.add_argument('-lr_encoder', type=float, default=3e-5)
    parser.add_argument('-lr_decoder', type=float, default=1e-4)
    parser.add_argument('-patience', type=int, default=3)
    parser.add_argument('-retrain', type=bool, default=False)

    opt = parser.parse_args()

    opt.device = 'cuda' if opt.no_cuda is False else 'cpu'
    if opt.device == 'cuda':
        assert torch.cuda.is_available()

    if opt.retrain:
        print('load checkpoint ...')
        checkpoint = torch.load('models/checkpoint.chkpt', map_location=torch.device(opt.device))
        # opt = checkpoint['settings']
    else:
        checkpoint = None

    data = pickle.load('data/m30k_deen_shr.pkl')

    src_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    vocab_trg = data['trg_vocab']

    utils.mkdir('models')
    pickle.dump(vocab_trg, 'models/trg_vocab.pkl')

    opt.src_pad = src_tokenizer.pad_token_id
    opt.trg_pad = vocab_trg.pad_idx

    opt.max_src_len = data['max_len']['src']
    opt.max_trg_len = data['max_len']['trg']

    train_data_loader, valid_data_loader, test_data_loader = prepare_dataloaders(opt, data)

    model = init_model(opt, vocab_trg.vocab_size, checkpoint=checkpoint)

    # optimizer for encoder
    param_optimizer = list(model.encoder.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer_encoder = AdamW(
        optimizer_grouped_parameters,
        lr=opt.lr_encoder,
        eps=1e-8
    )
    # Create the learning rate scheduler.
    total_steps = opt.batch_size * opt.epochs
    scheduler_encoder = get_linear_schedule_with_warmup(
        optimizer_encoder,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # optimizer for decoder
    optimizer_decoder = torch.optim.Adam(model.decoder.parameters(), lr=opt.lr_decoder, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(optimizer_decoder, T_max=len(train_data_loader))

    train(model, optimizer_encoder, scheduler_encoder, optimizer_decoder, train_data_loader, valid_data_loader, opt)

    test(model, test_data_loader, opt)


def prepare_dataloaders(opt, data):
    batch_size = opt.batch_size

    opt.trg_vocab_size = data['trg_vocab'].vocab_size

    train_inputs = torch.tensor(data['train']['src'])
    valid_inputs = torch.tensor(data['valid']['src'])
    test_inputs = torch.tensor(data['test']['src'])

    train_outputs = torch.tensor(data['train']['trg'])
    valid_outputs = torch.tensor(data['valid']['trg'])
    test_outputs = torch.tensor(data['test']['trg'])

    train_data = TensorDataset(train_inputs, train_outputs)
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    valid_data = TensorDataset(valid_inputs, valid_outputs)
    valid_sampler = SequentialSampler(valid_data)
    valid_data_loader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_inputs, test_outputs)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader

    # for asking about further training use while true loop, and return


if __name__ == "__main__":
    main()
