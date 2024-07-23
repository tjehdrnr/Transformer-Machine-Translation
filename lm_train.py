import argparse
import pprint

import torch
from torch import optim
import torch.nn as nn

from dataloader import DataLoader
import dataloader as dataloader

from models.rnnlm import LanguageModel
from lm_trainer import LanguageModelTrainer as LMTrainer

from dual_train import get_crits


def define_argparser(is_continue=False):
    p = argparse.ArgumentParser()

    if is_continue:
        p.add_argument(
            '--load_fn',
            required=True,
            help='Model file name to continue'
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
        help='Validation set file name except the extention. (ex: valid.en --> valid)'
    )
    p.add_argument(
        '--lang',
        required=not is_continue,
        help='Set of extention represents language pair. (ex: en + ko --> enko)'
    )
    p.add_argument(
        '--gpu_id',
        type=int,
        default=-1,
    )
    p.add_argument(
        '--off_autocast',
        action='store_true',
        help='Turn-off Automatic Mixed Precision(AMP), which speed-up training.'
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
        '--max_length',
        type=int,
        default=100,
        help='Maximum length of the training sequence. Default=%(default)s'
    )
    p.add_argument(
        '--dropout_p',
        type=float,
        default=.2,
    )
    p.add_argument(
        '--embedding_dim',
        type=int,
        default=512,
    )
    p.add_argument(
        '--hidden_size',
        type=int,
        default=768,
    )
    p.add_argument(
        '--n_layers',
        type=int,
        default=4,
    )
    p.add_argument(
        '--max_grad_norm',
        type=float,
        default=1e+8,
        help='Threshold for gradient clipping. Default=%(default)s'
    )

    config = p.parse_args()

    return config


def get_models(src_vocab_size, tgt_vocab_size, config):
    language_models = [
        LanguageModel(
            tgt_vocab_size,
            config.embedding_dim,
            config.hidden_size,
            n_layers=config.n_layers,
            dropout_p=config.dropout_p
        ),
        LanguageModel(
            src_vocab_size,
            config.embedding_dim,
            config.hidden_size,
            n_layers=config.n_layers,
            dropout_p=config.dropout_p
        )
    ]

    return language_models


def main(config):
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

    models = get_models(
        src_vocab_size,
        tgt_vocab_size,
        config
    )

    crits = get_crits(
        src_vocab_size,
        tgt_vocab_size,
        pad_index=dataloader.PAD
    )

    if config.gpu_id >= 0:
        for model, crit in zip(models, crits):
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)
    
    if config.verbose >= 2:
        print(models)
    
    for model, crit in zip(models, crits):
        optimizer = optim.Adam(model.parameters())
        lm_trainer = LMTrainer(config)

        model = lm_trainer.train(
            model, crit, optimizer,
            train_loader=loader.train_iter,
            valid_loader=loader.valid_iter,
            src_vocab=loader.src.vocab if model.vocab_size == src_vocab_size else None,
            tgt_vocab=loader.tgt.vocab if model.vocab_size == tgt_vocab_size else None,
            n_epochs=config.n_epochs
        )

    torch.save(
        {
            'model': [
                models[0].state_dict(),
                models[1].state_dict(),
            ],
            'config': config,
            'src_vocab': loader.src.vocab,
            'tgt_vocab': loader.tgt.vocab,
        }, config.model_fn
    )


if __name__ == '__main__':
    config = define_argparser()
    main(config)