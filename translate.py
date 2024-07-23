import argparse
import sys
import codecs
from operator import itemgetter

import torch

from dataloader import DataLoader
import dataloader as dataloader
from models.seq2seq import Seq2seq
from models.transformer import Transformer


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument(
        '--max_length', type=int, default=255,
        help='Maximum sequence length for inference. Default=%(default)s'
    )
    p.add_argument(
        '--n_best', type=int, default=1,
        help='Number of best inference result per sample. Default=%(default)s'
    )
    p.add_argument(
        '--beam_size', type=int, default=5,
        help='Beam size for beam search. Default=%(default)s'
    )
    p.add_argument(
        '--lang', type=str, default=None,
        help='Source language and target language. Example : enko'
    )
    p.add_argument(
        '--length_penalty', type=float, default=1.2,
        help='Length penalty parameter that higher value produce shorter results.'
    )

    config = p.parse_args()

    return config


def read_text(batch_size=128):
    lines = []

    sys.stdin = codecs.getreader('utf-8')(sys.stdin.detach())

    for line in sys.stdin:
        if line.strip() != '':
            lines.append(line.strip().split(' '))

            if len(lines) >= batch_size:
                yield lines
                lines = []
        
    if len(lines) > 0:
        yield lines


def to_text(indice, vocab):
    lines = []

    for i in range(len(indice)):
        line = []
        for j in range(len(indice[i])):
            index = indice[i][j]

            if index == dataloader.EOS:
                break
            else:
                line.append(vocab.itos[index])
        
        line = ' '.join(line)
        lines.append(line)
    
    return lines


def is_dsl(train_config):
    return 'dsl_lambda' in vars(train_config).keys()
    # return not ('rl_n_epochs' in vars(train_config).keys())


def get_vocabs(train_config, config, saved_data):
    if is_dsl(train_config):
        assert config.lang is not None
    
        if config.lang == train_config.lang:
            is_reverse = False
        else:
            is_reverse = True
        
        if not is_reverse:
            # Load vocabularies from the model.
            src_vocab = saved_data['src_vocab']
            tgt_vocab = saved_data['tgt_vocab']
        else:
            src_vocab = saved_data['tgt_vocab']
            tgt_vocab = saved_data['src_vocab']
        
        return src_vocab, tgt_vocab, is_reverse
    else:
        # Load vocabularies from the model.
        src_vocab = saved_data['src_vocab']
        tgt_vocab = saved_data['tgt_vocab']

    return src_vocab, tgt_vocab, False


def get_model(input_size, output_size, train_config, is_reverse=False):
    # Declare sequence-to-sequence model.
    if 'use_transformer' in vars(train_config).keys() and train_config.use_transformer:
        model = Transformer(
            input_size,
            train_config.hidden_size,
            output_size,
            n_heads=train_config.n_heads,
            n_enc_blocks=train_config.n_layers,
            n_dec_blocks=train_config.n_layers,
            dropout_p=train_config.dropout_p
        )
    else:
        model = Seq2seq(
            input_size,
            train_config.embedding_dim,
            train_config.hidden_size,
            output_size,
            n_layers=train_config.n_layers,
            dropout_p=train_config.dropout_p
        )
    
    if is_dsl(train_config):
        if not is_reverse:
            model.load_state_dict(saved_data['model'][0])
        else:
            model.load_state_dict(saved_data['model'][1])
    else:
        model.load_state_dict(saved_data['model'])
    model.eval()

    return model


if __name__ == '__main__':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    config = define_argparser()

    # Load saved data.
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )

    # Load configuration setting in training.
    train_config = saved_data['config']

    src_vocab, tgt_vocab, is_reverse = get_vocabs(train_config, config, saved_data)

    loader = DataLoader()
    loader.load_vocab(src_vocab, tgt_vocab)
    
    input_size, output_size = len(loader.src.vocab), len(loader.tgt.vocab)
    model = get_model(input_size, output_size, train_config, is_reverse)

    # Put models to device if it is necessary.
    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
    
    with torch.no_grad():
        # Get sentences from standard input.
        for lines in read_text(batch_size=config.batch_size):
            lengths = [len(line) for line in lines]
            original_indice = [i for i in range(len(lines))]

            sorted_tuples = sorted(
                zip(lines, lengths, original_indice),
                key=itemgetter(1),
                reverse=True
            )
            sorted_lines = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]
            lengths = [sorted_tuples[i][1] for i in range(len(sorted_tuples))]
            sorted_indice = [sorted_tuples[i][2] for i in range(len(sorted_tuples))]

            # Converts string to list of index.
            x = loader.src.numericalize(
                loader.src.pad(sorted_lines),
                device='cuda:%d' % config.gpu_id if config.gpu_id >= 0 else 'cpu'
            )
            # |x| = (batch_size, length)

            if config.beam_size == 1:
                y_hat, indice = model.search(x)
                output = to_text(indice, loader.tgt.vocab)

                sorted_tuples = sorted(zip(output, original_indice), key=itemgetter(1))
                ouput = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]

                sys.stdout.write('\n'.join(output) + '\n')
            else:
                batch_indice, _ = model.batch_beam_search(
                    x,
                    beam_size=config.beam_size,
                    max_length=config.max_length,
                    n_best=config.n_best,
                    length_penalty=config.length_penalty
                )
                
                # Restore the original indice.
                output = []
                for i in range(len(batch_indice)):
                    output.append(to_text(batch_indice[i], loader.tgt.vocab))
                sorted_tuples = sorted(zip(output, original_indice), key=itemgetter(1))
                output = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]

                for i in range(len(output)):
                    sys.stdout.write('\n'.join(output[i]) + '\n')
