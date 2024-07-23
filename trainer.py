import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as torch_utils
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from utils import get_grad_norm, get_parameter_norm


VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class MaximumLikelihoodEstimationEngine(Engine):

    def __init__(self, func, model, crit, optimizer, lr_scheduler, config):
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config

        super().__init__(func)

        self.best_loss = float('inf')
        self.scaler = GradScaler()
    
    @staticmethod
    def train(engine, mini_batch):
        engine.model.train()

        # Gradient accumulation.
        if engine.state.iteration % engine.config.iteration_per_update == 1 or \
            engine.config.iteration_per_update == 1:
              engine.optimizer.zero_grad()

        device = next(engine.model.parameters()).device
        mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1])
        mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1])

        x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
        # |x| = (batch_size, n)
        # |y| = (batch_size, m)

        with autocast(): # Automatic Mixed Precision.
            y_hat = engine.model(x, mini_batch.tgt[0][:, :-1])
            # |y_hat| = (batch_size, m, output_size)
            
            loss = engine.crit(
                y_hat.contiguous().view(-1, y_hat.size(-1)),
                y.contiguous().view(-1)
            )
            # Because of NLLLoss's reduction is 'sum',
            backward_target = loss.div(y.size(0)).div(engine.config.iteration_per_update)
        
        if engine.config.gpu_id >= 0:
            engine.scaler.scale(backward_target).backward()
        else:
            backward_target.backward()

        word_count = int(mini_batch.tgt[1].sum())
        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        if engine.state.iteration % engine.config.iteration_per_update == 0:
            # In order to avoid gradient exploding, we apply gradient clipping.
            torch_utils.clip_grad_norm_(
                engine.model.parameters(),
                engine.config.max_grad_norm,
            )
            # Take a step of gradient descent.
            if engine.config.gpu_id >= 0:
                engine.scaler.step(engine.optimizer)
                engine.scaler.update()
            else:
                engine.optimizer.step()
            
            # if engine.config.use_noam_decay and engine.lr_scheduler is not None:
            #     engine.lr_scheduler.step()
        
        y_hat_ = y_hat.contiguous().view(-1, y_hat.size(-1))
        y_ = y.contiguous().view(-1)

        loss = float(loss / word_count) # Calculate each word's loss.
        ppl = np.exp(loss)

        return {
            'loss': loss,
            'ppl': ppl,
            '|param|': p_norm if not np.isnan(p_norm) and not np.isinf(p_norm) else 0.,
            '|g_param|': g_norm if not np.isnan(g_norm) and not np.isinf(g_norm) else 0.
        }
    
    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            device = next(engine.model.parameters()).device
            
            mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1])
            mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1])

            x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
            # |x| = (batch_size, n)
            # |y| = (batch_size, m)

            with autocast():
                y_hat = engine.model(x, mini_batch.tgt[0][:, :-1])
                # |y_hat| = (batch_size, m, output_size)

                loss = engine.crit(
                    y_hat.contiguous().view(-1, y_hat.size(-1)),
                    y.contiguous().view(-1)
                )

            word_count = int(mini_batch.tgt[1].sum())
            y_hat_ = y_hat.contiguous().view(-1, y_hat.size(-1))
            y_ = y.contiguous().view(-1)

            loss = float(loss / word_count)
            ppl = np.exp(loss)

            return {
                'loss': loss,
                'ppl': ppl
            }
    
    @staticmethod
    def attach(
        train_engine, validation_engine,
        training_metric_names = ['loss', 'ppl', '|param|', '|g_param|'],
        validation_metric_names = ['loss', 'ppl'],
        verbose=VERBOSE_EPOCH_WISE
    ):

        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name
            )
        
        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)
        
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names[1:])
        
        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                avg_p_norm = engine.state.metrics['|param|']
                avg_g_norm = engine.state.metrics['|g_param|']
                avg_loss = engine.state.metrics['loss']
                avg_ppl = np.exp(avg_loss)

                print('Epoch {} / {}'.format(engine.state.epoch, engine.config.n_epochs))
                print('Train - |param|: {:.4e}  |g_param|: {:.4e} loss: {:.4e}  ppl: {:.2f}'.format(
                    avg_p_norm,
                    avg_g_norm,
                    avg_loss,
                    avg_ppl,
                ))
        
        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)
        
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names[1:])
        
        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                avg_loss = engine.state.metrics['loss']
                avg_ppl = np.exp(avg_loss)

                print('Valid - loss: {:.4e} ppl: {:.2f} best_loss: {:.4e} best_ppl: {:.2f}'.format(
                    avg_loss,
                    avg_ppl,
                    engine.best_loss,
                    np.exp(engine.best_loss)
                ))
    
    @staticmethod
    def resume_training(engine, resume_epoch):
        engine.state.iteration = (resume_epoch - 1) * len(engine.state.dataloader)
        engine.state.epoch = (resume_epoch - 1)

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            print(f"best_loss가 {engine.best_loss:.4f}에서 {loss:.4f}로 변경됨.")
            engine.best_loss = loss
            
    
    @staticmethod
    def save_model(engine, train_engine, config, src_vocab, tgt_vocab):
        avg_train_loss = train_engine.state.metrics['loss']
        avg_valid_loss = engine.state.metrics['loss']

        model_fn = config.model_fn.split('.')

        model_fn = model_fn[:-1] + [
            '%02d' % train_engine.state.epoch,
            '%.2f-%.2f' % (avg_train_loss, np.exp(avg_train_loss)),
            '%.2f-%.2f' % (avg_valid_loss, np.exp(avg_valid_loss))
            ] + [model_fn[-1]]
        
        model_fn = '.'.join(model_fn)

        torch.save(
            {
                'model': engine.model.state_dict(),
                'opt': train_engine.optimizer.state_dict(),
                'config': config,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab
            }, model_fn
        )


class SingleTrainer():

    def __init__(self, target_engine_class, config):
        self.target_engine_class = target_engine_class
        self.config = config

    def train(
        self,
        model, crit, optimizer,
        train_loader, valid_loader,
        src_vocab, tgt_vocab,
        n_epochs,
        lr_scheduler=None
    ):
        train_engine = self.target_engine_class(
            self.target_engine_class.train,
            model,
            crit,
            optimizer,
            lr_scheduler,
            self.config
        )
        validation_engine = self.target_engine_class(
            self.target_engine_class.validate,
            model,
            crit,
            optimizer=None,
            lr_scheduler=None,
            config=self.config
        )

        self.target_engine_class.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

            if engine.lr_scheduler is not None and not engine.config.use_noam_decay:
                engine.lr_scheduler.step()
        
        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED, run_validation, validation_engine, valid_loader
        )
        train_engine.add_event_handler(
            Events.STARTED,
            self.target_engine_class.resume_training,
            self.config.init_epoch
        )

        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, 
            self.target_engine_class.check_best,
            train_engine,
            self.config,
            src_vocab,
            tgt_vocab
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            self.target_engine_class.save_model,
            train_engine,
            self.config,
            src_vocab,
            tgt_vocab
        )

        train_engine.run(train_loader, max_epochs=n_epochs)

        return model

      
            

        


        


