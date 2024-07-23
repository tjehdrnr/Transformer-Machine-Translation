import argparse
import pprint

import torch
import torch.nn as nn
from torch import optim

from transformers import get_linear_schedule_with_warmup

from dataloader import DataLoader
import dataloader as dataloader
from models.seq2seq import Seq2seq
from models.transformer import Transformer
from trainer import MaximumLikelihoodEstimationEngine, SingleTrainer
from rl_trainer import MinimumRiskTrainingEngine


def define_argparser(is_continue=False):
    p = argparse.ArgumentParser()

    if is_continue:
        p.add_argument(
            '--load_fn',
            required=True,
            help='Model file name to continue.'
        )

    p.add_argument(
        '--model_fn',
        required=not is_continue,
        help='Model file name to save. Additional information would be annotated to the file name.'
    )
    p.add_argument(
        '--train',
        required=not is_continue,
        help='Training set file name except the extension. (ex: train.en --> train)'
    )
    p.add_argument(
        '--valid',
        required=not is_continue,
        help='Validation set file name except the extension. (ex: valid.en --> valid)'
    )
    p.add_argument('--lang', type=str, required=not is_continue)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--off_autocast', action='store_true')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument(
        '--init_epoch', 
        type=int, 
        required=is_continue, 
        default=1,
        help='Set initial epoch number, which can be useful in continue training.'
    )

    p.add_argument(
        '--max_length',
        type=int,
        default=100,
        help='Maximum length of the training sequence.'
    )
    p.add_argument('--dropout_p', type=float, default=.2)
    p.add_argument('--embedding_dim', type=int, default=512)
    p.add_argument('--hidden_size', type=int, default=768)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--max_grad_norm', type=float, default=5.)
    p.add_argument('--iteration_per_update', type=int, default=1)

    p.add_argument('--lr', type=float, default=1.)
    p.add_argument('--lr_step', type=int, default=1)
    p.add_argument('--lr_gamma', type=float, default=.5)
    p.add_argument('--lr_decay_start', type=int, default=10)
    p.add_argument('--use_adam', action='store_true')
    p.add_argument('--use_radam', action='store_true')

    p.add_argument('--use_transformer', action='store_true')
    p.add_argument('--n_heads', type=int, default=8)

    p.add_argument(
        '--rl_lr',
        type=float,
        default=.01,
        help='Learning rate for reinforcement learning. Default=%(default)s'
    )
    p.add_argument(
        '--rl_n_samples',
        type=int,
        default=1,
        help='Number of samples to get baseline. Default=%(default)s'
    )
    p.add_argument(
        '--rl_n_epochs',
        type=int,
        default=10,
        help='Number of epochs for reinforcement learning. Default=%(default)s'
    )
    p.add_argument(
        '--rl_n_gram',
        type=int,
        default=6,
        help='Maximum number of tokens to calculate BLEU for reinforcement learning.'
    )
    p.add_argument(
        '--rl_reward',
        type=str,
        default='gleu',
        help='Method name to use as reward function for RL training. Default=%(default)s'
    )
    
    config = p.parse_args()

    return config


def get_model(input_size, output_size, config):
    if config.use_transformer:
        model = Transformer(
            input_size, # Source vocabulary size
            config.hidden_size, # Transformer doesn't need embedding dim.
            output_size,  # Target vocabulary size
            n_heads=config.n_heads,  # Number of head in Multi-head attention.
            n_enc_blocks=config.n_layers,
            n_dec_blocks=config.n_layers,
            dropout_p=config.dropout_p
        )
    else:
        model = Seq2seq(
            input_size,
            config.embedding_dim,
            config.hidden_size,
            output_size,
            n_layers=config.n_layers,
            dropout_p=config.dropout_p
        )

    return model


def get_crit(output_size, pad_index):
    loss_weight = torch.ones(output_size)
    loss_weight[pad_index] = 0.

    crit = nn.NLLLoss(
        weight=loss_weight,
        reduction='sum'
    )

    return crit


def get_optimizer(model, config):
    if config.use_adam:
        if config.use_transformer:
            optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(.9, .98))
        else: # Case of rnn based seq2seq.
            optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.use_radam:
        optimizer = optim.RAdam(model.parameters(), lr=config.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr)
    
    return optimizer


def get_scheduler(optimizer, n_minibatchs, config):
    if config.lr_step > 0:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[i for i in range(
                max(0, config.lr_decay_start - 1),
                (config.init_epoch - 1) + config.n_epochs,
                config.lr_step
            )],
            gamma=config.lr_gamma,
            last_epoch=config.init_epoch - 1 if config.init_epoch > 1 else -1,
        )
    else:
        lr_scheduler = None
    
    return lr_scheduler


def main(config, model_weight=None, opt_weight=None):
    def print_config(config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)

    loader = DataLoader(
        config.train,
        config.valid,
        (config.lang[:2], config.lang[2:]),
        batch_size=config.batch_size,
        device=-1,
        max_length=config.max_length,
        dsl=False
    )

    input_size, output_size = len(loader.src.vocab), len(loader.tgt.vocab)
    model = get_model(input_size, output_size, config)
    crit = get_crit(output_size, dataloader.PAD)

    if model_weight is not None:
        model.load_state_dict(model_weight)
    
    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)
    
    optimizer = get_optimizer(model, config)

    if opt_weight is not None and (config.use_adam or config.use_radam):
        optimizer.load_state_dict(opt_weight)
    
    lr_scheduler = get_scheduler(optimizer, len(loader.train_iter), config)

    if config.verbose >= 2:
        print(model)
        print(crit)
        print(optimizer)
    
    mle_trainer = SingleTrainer(MaximumLikelihoodEstimationEngine, config)
    mle_trainer.train(
        model,
        crit,
        optimizer,
        train_loader=loader.train_iter,
        valid_loader=loader.valid_iter,
        src_vocab=loader.src.vocab,
        tgt_vocab=loader.tgt.vocab,
        n_epochs=config.n_epochs,
        lr_scheduler=lr_scheduler
    )

    if config.rl_n_epochs > 0:
        optimizer = optim.SGD(model.parameters(), lr=config.rl_lr)
        mrt_trainer = SingleTrainer(MinimumRiskTrainingEngine, config)

        mrt_trainer.train(
            model,
            None, # We don't need criterion for MRT.
            optimizer,
            train_loader=loader.train_iter,
            valid_loader=loader.valid_iter,
            src_vocab=loader.src.vocab,
            tgt_vocab=loader.tgt.vocab,
            n_epochs=config.rl_n_epochs
        )


if __name__ == '__main__':
    config = define_argparser()
    main(config)
