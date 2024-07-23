from operator import itemgetter

import torch
import torch.nn as nn

import dataloader as dataloader

LENGTH_PENALTY = .2
MIN_LENGTH = 5


class SingleBeamSearchBoard():

    def __init__(
        self,
        device,
        prev_status_config,
        beam_size=5,
        max_length=255
    ):
        self.beam_size = beam_size
        self.max_length = max_length

        self.device = device
        # Inferred word index for each time-step. For now, initiallized with initial time-step
        self.word_indice = [torch.LongTensor(beam_size).zero_().to(self.device) + dataloader.BOS]
        # Beam index for selected word index, at each time-step.
        self.beam_indice = [torch.LongTensor(beam_size).zero_().to(self.device) - 1]
        # Cumulative log-probability for each beam.
        self.cumulative_probs = [torch.FloatTensor([.0] + [-float('inf')] * (beam_size - 1)).to(self.device)]
        # 1 if it is done else 0
        self.masks = [torch.BoolTensor(beam_size).zero_().to(self.device)]

        self.prev_status = {}
        self.batch_dims = {}
        for prev_status_name, each_config in prev_status_config.items():
            init_status = each_config['init_status']
            batch_dim_index = each_config['batch_dim_index']
            if init_status is not None:
                self.prev_status[prev_status_name] = torch.cat([init_status] * beam_size,
                                                               dim=batch_dim_index)
            else:
                self.prev_status[prev_status_name] = None
            self.batch_dims[prev_status_name] = batch_dim_index
            
        self.current_time_step = 0
        self.done_cnt = 0

    def is_done(self):
        # Return 1, if we had EOS more than 'beam_size'-times.
        if self.done_cnt >= self.beam_size:
            return 1
        return 0

    def get_length_penalty(
        self,
        length,
        alpha=LENGTH_PENALTY,
        min_length=MIN_LENGTH,
    ):
        # Calculate length-penalty,
        # because shorter sentence usually have bigger probability.
        # In fact, we represent this as log-probability, which is negative value.
        # Thus, we need to multiply bigger penalty for shorter one.
        p = ((min_length + 1) / (min_length + length))**alpha

        return p
    
    def get_batch(self):
        y_hat = self.word_indice[-1].unsqueeze(-1)
        # |y_hat| = (beam_size, 1)
        # if model != transformer:
        #     |hidden|, |cell| = (n_layers, beam_size, hidden_size)
        #     |h_t_tilde| = (beam_size, 1, hidden_size) or None

        return y_hat, self.prev_status
    
    def collect_result(self, y_hat, prev_status):
        output_size = y_hat.size(-1)

        self.current_time_step += 1

        cumulative_prob = self.cumulative_probs[-1].masked_fill_(self.masks[-1], -float('inf'))
        cumulative_prob = y_hat + cumulative_prob.view(-1, 1, 1).expand(self.beam_size, 1, output_size)
        # |cumulative_prob| = (beam_size, 1, output_size)
        
        top_log_prob, top_indice = torch.topk(
            cumulative_prob.view(-1), # (beam_size * output_size,)
            self.beam_size,
            dim=-1
        )
        # |top_log_prob| = (beam_size,)
        # |top_indice| = (beam_size,)

        self.word_indice += [top_indice.fmod(output_size)]
        self.beam_indice += [top_indice.div(float(output_size)).long()]

        self.cumulative_probs += [top_log_prob]
        self.masks += [torch.eq(self.word_indice[-1], dataloader.EOS)]
        # Caluate a number of finished beams.
        self.done_cnt += self.masks[-1].float().sum()

        for prev_status_name, prev_status in prev_status.items():
            self.prev_status[prev_status_name] = torch.index_select(
                prev_status, # (n_layers, beam_size, hidden_size)
                dim=self.batch_dims[prev_status_name],
                index=self.beam_indice[-1]
            ).contiguous()
    
    def get_n_best(self, n=1, length_penalty=.2):
        sentences, probs, founds = [], [], []

        for t in range(len(self.word_indice)):
            for b in range(self.beam_size):
                if self.masks[t][b] == 1:
                    probs += [self.cumulative_probs[t][b] * self.get_length_penalty(
                        t,
                        alpha=length_penalty
                    )]
                    founds += [(t, b)]
        # Also, collect log-probability from last time-step, for the case of EOS is not shown.
        for b in range(self.beam_size):
            if self.cumulative_probs[-1][b] != -float('inf'):
                if not (len(self.cumulative_probs) - 1, b) in founds:
                    probs += [self.cumulative_probs[-1][b] *
                      self.get_length_penalty(len(self.cumulative_probs), alpha=length_penalty)]
                    founds += [(t, b)]
        
        sorted_founds_with_probs = sorted(
            zip(founds, probs),
            key=itemgetter(1),
            reverse=True
        )[:n]
        probs = []

        for (end_index, b), prob in sorted_founds_with_probs:
            sentence = []

            # Trace from the end.
            for t in range(end_index, 0, -1):
                sentence = [self.word_indice[t][b]] + sentence
                b = self.beam_indice[t][b]
            
            sentences += [sentence]
            probs += [prob]
        
        return sentences, probs
                    

        