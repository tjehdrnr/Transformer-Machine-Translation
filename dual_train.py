import argparse
import pprint

import torch
from torch import optim
import torch.nn as nn

from dataloader import DataLoader
import dataloader as dataloader

from models.seq2seq import Seq2seq
from models.transformer import Transformer
from models.rnnlm import LanguageModel

from lm_trainer import LanguageModelTrainer as LMTrainer
from dual_trainer import DualSupervisedTrainer as DSLTrainer


def define_argparser(is_continue=False):
    p = argparse.ArgumentParser()

    if is_continue:
        p.add_argument(
            '--load_fn',
            required=True
        )
    
    p.add_argument(
        '--model_fn',
        required=not is_continue
    )
    p.add_argument(
        '--lm_fn',
        required=not is_continue
    )
    p.add_argument(
        '--train',
        required=not is_continue
    )
    p.add_argument(
        '--valid',
        required=not is_continue
    )
    p.add_argument(
        '--lang',
        required=not is_continue
    )
    p.add_argument(
        '--gpu_id',
        type=int,
        default=-1
    )
    p.add_argument(
        '--off_autocast',
        action='store_true'
    )
    p.add_argument(
        '--batch_size',
        type=int,
        default=32
    )
    p.add_argument(
        '--n_epochs',
        type=int,
        default=20
    )
    p.add_argument(
        '--verbose',
        type=int,
        default=2
    )
    p.add_argument(
        '--init_epoch',
        type=int,
        default=-1
    )
    p.add_argument(
        '--max_length',
        type=int,
        default=100
    )
    p.add_argument(
        '--dropout_p',
        type=float,
        default=.2
    )
    p.add_argument(
        '--embedding_dim',
        type=int,
        default=512
    )
    p.add_argument(
        '--n_layers',
        type=int,
        default=4
    )
    p.add_argument(
        '--max_grad_norm',
        type=float,
        default=1e+8
    )
    p.add_argument(
        '--iteration_per_update',
        type=int,
        default=1
    )

    p.add_argument(
        '--dsl_n_warmup_epochs',
        type=int,
        default=2
    )
    p.add_argument(
      '--dsl_lambda',
      type=float,
      default=1e-3,
      help='Lagrangian Multiplier for regularization term.'
    )
    
    p.add_argument(
        '--use_transformer',
        action='store_true'
    )
    p.add_argument(
        '--n_heads',
        type=int,
        default=8
    )

    config = p.parse_args()

    return config


def load_lm(fn, language_models):
    saved_data = torch.load(fn, map_location='cpu')

    model_weight = saved_data['model']
    language_models[0].load_state_dict(model_weight[0])
    language_models[1].load_state_dict(model_weight[1])


def get_models(src_vocab_size, tgt_vocab_size, config):
    language_models = [ # X2Y
        LanguageModel(
            tgt_vocab_size,
            config.embedding_dim,
            config.hidden_size,
            n_layers=config.n_layers,
            dropout_p=config.dropout_p
        ),
        LanguageModel( # Y2X
            src_vocab_size,
            config.embedding_dim,
            config.hidden_size,
            n_layers=config.n_layers,
            dropout_p=config.dropout_p
        )
    ]

    if config.use_transformer:
        models = [
            Transformer( # X2Y
                src_vocab_size,
                config.hidden_size,
                tgt_vocab_size,
                n_heads=config.n_heads,
                n_enc_blocks=config.n_layers,
                n_dec_blocks=config.n_layers,
                dropout_p=config.dropout_p
            ),
            Transformer( # Y2X
                tgt_vocab_size,
                config.hidden_size,
                src_vocab_size,
                n_heads=config.n_heads,
                n_enc_blocks=config.n_layers,
                n_dec_blocks=config.n_layers,
                dropout_p=config.dropout_p
            )
        ]
    else:
        models = [
            Seq2seq( # X2Y
                src_vocab_size,
                config.embedding_dim,
                config.hidden_size,
                tgt_vocab_size,
                n_layers=config.n_layers,
                dropout_p=config.dropout_p
            ),
            Seq2seq( # Y2X
                tgt_vocab_size,
                config.embedding_dim,
                config.hidden_size,
                src_vocab_size,
                n_layers=config.n_layers,
                dropout_p=config.dropout_p
            )
        ]
    
    return language_models, models


def get_crits(src_vocab_size, tgt_vocab_size, pad_index):
    loss_weights = [
        torch.ones(tgt_vocab_size),
        torch.ones(src_vocab_size)
    ]
    loss_weights[0][pad_index] = .0
    loss_weights[1][pad_index] = .0

    crits = [
        nn.NLLLoss(weight=loss_weights[0], reduction='none'),
        nn.NLLLoss(weight=loss_weights[1], reduction='none')
    ]

    return crits


def get_optimizers(models, config):
    if config.use_transformer:
        optimizers = [
            optim.Adam(models[0].parameters(), betas=(.9, .98)),
            optim.Adam(models[1].parameters(), betas=(.9, .98))
        ]
    else:
        optimizers = [
            optim.Adam(models[0].parameters()),
            optim.Adam(models[1].parameters())
        ]
    
    return optimizers


def main(config, model_weight=None, opt_weight=None):
    def print_config(config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)

    loader = DataLoader(
        config.train,
        config.valid,
        (config.lang[:2], config.lang[-2:]),
        batch_size=config.batch_size,
        device=-1,
        max_length=config.max_length,
        dsl=True
    )

    src_vocab_size = len(loader.src.vocab)
    tgt_vocab_size = len(loader.tgt.vocab)

    language_models, models = get_models(
        src_vocab_size,
        tgt_vocab_size,
        config
    )

    crits = get_crits(
        src_vocab_size,
        tgt_vocab_size,
        pad_index=dataloader.PAD
    )

    if model_weight is not None:
        for model, w in zip(models + language_models, model_weight):
            model.load_state_dict(w)
    
    load_lm(config.lm_fn, language_models)

    if config.gpu_id >= 0:
        for lm, seq2seq, crit in zip(language_models, models, crits):
            lm.cuda(config.gpu_id)
            seq2seq.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)
    
    dsl_trainer = DSLTrainer(config)

    optimizers = get_optimizers(models, config)

    if opt_weight is not None:
        for opt, w in zip(optimizers, opt_weight):
            opt.load_state_dict(w)
    
    if config.verbse >= 2:
        print(language_models)
        print(models)
        print(crits)
        print(optimizers)
    
    dsl_trainer.train(
        models,
        language_models,
        crits,
        optimizers,
        train_loader=loader.train_iter,
        valid_loader=loader.valid_iter,
        vocabs=[loader.src.vocab, loader.tgt.vocab],
        n_epochs=config.n_epochs,
        lr_schedulers=None
    )   


if __name__ == '__main__':
    config = define_argparser()
    main(config)