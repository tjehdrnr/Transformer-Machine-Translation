import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import dataloader as dataloader
from search import SingleBeamSearchBoard


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers=4, dropout_p=.2):

        self.input_size = input_size # embedding_dim
        self.hidden_size = int(hidden_size / 2)
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super(Encoder, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=True,
        )

    def forward(self, emb):
        # |x| = (batch_size, n, embedding_dim)

        if isinstance(emb, tuple):
            x, lengths = emb
            x = pack(x, lengths.tolist(), batch_first=True)
        else:
            x = emb

        y, h = self.rnn(x)
        # |y| = (batch_size, n, hidden_size)
        # |h[0], h[1]| = (2 * n_layers, batch_size, hidden_size / 2)

        if isinstance(emb, tuple):
            y, _ = unpack(y, batch_first=True)
        
        return y, h


class Attention(nn.Module):

    def __init__(self, hidden_size):

        super(Attention, self).__init__()

        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, h_src, h_t_tgt, mask=None):
        # |h_t| = (batch_size, 1, hidden_size)
        # |enc_hiddens| = (batch_size, n, hidden_size)
        # |mask| = (batch_size, n)

        query = self.linear(h_t_tgt)
        # |query| = (batch_size, 1, hidden_size)
        attn_scores = torch.bmm(query, h_src.transpose(1, 2))
        # |attn_scores| = (batch_size, 1, n)

        if mask is not None:
            attn_scores.masked_fill(mask.unsqueeze(1), -float('inf'))

        attn_weights = self.softmax(attn_scores)
        # |attn_weigths| = (batch_size, 1, n)
        context_vector = torch.bmm(attn_weights, h_src)
        # |context_vector| = (batch_size, 1, hidden_size)

        return context_vector


class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers=4, dropout_p=.2):

        self.input_size = input_size # embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super(Decoder, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size + hidden_size, # Input shape for Input-feeding.
            hidden_size=hidden_size, 
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=False,
        )
    
    def forward(self, emb_t, h_t_1_tilde, h_t_1):
        # |emb_t| = (batch_size, 1, embedding_dim)
        # |h_t_1_tilde| = (batch_size, 1, hidden_size)
        # |h_t_1[0], h_t_1[1]| = (n_layers, batch_size, hidden_size)
        batch_size = emb_t.size(0)
        hidden_size = h_t_1[0].size(-1)

        if h_t_1_tilde is None:
            h_t_1_tilde = emb_t.new_zeros(batch_size, 1, hidden_size)

        x = torch.cat([emb_t, h_t_1_tilde], dim=-1) # Input-feeding trick.

        y, h = self.rnn(x, h_t_1)
        # |h_t_tgt| = (batch_size, 1, hidden_size)
        # |h_t_1[0], h_t_1[1]| = (n_layers, batch_size, hidden_size)

        return y, h


class Generator(nn.Module):

    def __init__(self, hidden_size, output_size):

        self.hidden_size = hidden_size
        self.output_size = output_size # Length of target vocabulary

        super(Generator, self).__init__()

        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x):
        # |x| = (batch_size, m, hidden_size)

        y_hats = self.softmax(self.output(x))
        # |y_hats| = (batch_size, m, output_size)

        return y_hats


class Seq2seq(nn.Module):

    def __init__(self, input_size, embedding_dim, hidden_size, output_size, n_layers=4, dropout_p=.2):
        
        self.input_size = input_size # src_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size # tgt_vocab_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super(Seq2seq, self).__init__()

        self.emb_src = nn.Embedding(input_size, embedding_dim)
        self.emb_tgt = nn.Embedding(output_size, embedding_dim)

        self.encoder = Encoder(embedding_dim, hidden_size, n_layers, dropout_p)
        self.decoder = Decoder(embedding_dim, hidden_size, n_layers, dropout_p)

        self.attn = Attention(hidden_size)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.generator = Generator(hidden_size, output_size)
    

    def merge_encoder_hiddens(self, h_0_tgt):
        # |h_0_tgt[0]| = (2 * n_layers, batch_size, hidden_size / 2)           
        new_hiddens, new_cells = [], []
        hiddens, cells = h_0_tgt

        for i in range(0, hiddens.size(0), 2):
            new_hiddens.append(torch.cat([hiddens[i], hiddens[i+1]], dim=-1))
            new_cells.append(torch.cat([cells[i], cells[i+1]], dim=-1))
            # |new_hidden, new_cell| = (1, batch_size, hidden_size)
        h_0_tgt = torch.stack(new_hiddens, dim=0)
        c_0_tgt = torch.stack(new_cells, dim=0)

        return (h_0_tgt, c_0_tgt)
    

    def fast_merge_encoder_hiddens(self, encoder_hiddens):
        h_0_tgt, c_0_tgt = encoder_hiddens
        batch_size = h_0_tgt.size(1)

        h_0_tgt = h_0_tgt.transpose(0, 1).contiguous().view(
            batch_size,
            -1,
            self.hidden_size
          ).transpose(0, 1).contiguous()
        
        c_0_tgt = c_0_tgt.transpose(0, 1).contiguous().view(
            batch_size,
            -1,
            self.hidden_size
          ).transpose(0, 1).contiguous()
        
        return (h_0_tgt, c_0_tgt)


    def generate_mask(self, x, length):
        # |x| = (batch_size, n)
        mask = []

        max_length = max(length)
        for l in length:
            if max_length - l > 0:
                mask.append(torch.cat([
                    x.new_zeros(1, l),
                    x.new_ones(1, (max_length - l))
                ], dim=-1))
            else:
                mask.append(x.new_zeros(1, l))

        mask = torch.cat(mask, dim=0).bool()
        
        return mask
    
    # def generate_mask(self, x, lengths):
    #     batch_size = x.size(0)

    #     mask = x.new_ones(batch_size, max(lengths))
    #     for i, length in enumerate(lengths):
    #         mask[i, :length] = x.new_zeros(1, length)
        
    #     return mask.bool()
    
    def forward(self, src, tgt):
        # |src| = (batch_size, n)
        # |tgt| = (batch_size, m)
        mask = None
        x_length = None
        if isinstance(src, tuple):
            x, x_length = src
            mask = self.generate_mask(x, x_length)
        else:
            x = src
        
        if isinstance(tgt, tuple):
            tgt = tgt[0]

        emb_src = self.emb_src(x)
        # |emb_src| = (batch_size, n, embedding_dim)
      
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        # |h_src| = (batch_size, n, hidden_size)
        # |h_0_tgt| = (2 * n_layers, batch_size, hidden_size / 2)
        h_0_tgt = self.fast_merge_encoder_hiddens(h_0_tgt)
        # |h_0_tgt| = (n_layers, batch_size, hidden_size)
        emb_tgt = self.emb_tgt(tgt)
        # |emb_tgt| = (batch_size, m, embedding_dim)

        h_tilde = []

        h_t_tilde = None
        decoder_hidden = h_0_tgt
        for t in range(tgt.size(1)):
            emb_t = emb_tgt[:, t, :].unsqueeze(1)
            # |emb_t| = (batch_size, 1, embedding_dim)
            decoder_output, decoder_hidden = self.decoder(emb_t, h_t_tilde, decoder_hidden)
            # |decoder_output| = (batch_size, 1, hidden_size)
            context_vector = self.attn(h_src, decoder_output, mask)
            # |context_vector| = (batch_size, 1, hidden_size)
            h_t_tilde = self.tanh(self.concat(torch.cat([decoder_output, context_vector], dim=-1)))
            # |h_t_1_tilde| = (batch_size, 1, hidden_size)
            h_tilde.append(h_t_tilde)
        
        h_tilde = torch.cat(h_tilde, dim=1)
        # |h_tilde| = (batch_size, m, hidden_size)
        
        y_hat = self.generator(h_tilde)
        # |y_hat| = (batch_size, m, output_size)

        return y_hat
    

    def search(self, src, is_greedy=True, max_length=255):
        if isinstance(src, tuple):
            x, x_length = src
            mask = self.generate_mask(x, x_length)
        else:
            x, x_length = src, None
            mask = None
        batch_size = x.size(0)

        emb_src = self.emb_src(x)
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        decoder_hidden = self.fast_merge_encoder_hiddens(h_0_tgt)

        y = x.new(batch_size, 1).zero_() + dataloader.BOS

        is_decoding = x.new_ones(batch_size, 1).bool()
        h_t_tilde, y_hats, indice = None, [], []

        while is_decoding.sum() > 0 and len(indice) < max_length:

            emb_t = self.emb_tgt(y)
            # |emb_t| = (batch_size, 1, embedding_dim)

            decoder_output, decoder_hidden = self.decoder(emb_t, h_t_tilde, decoder_hidden)
            context_vector = self.attn(h_src, decoder_output, mask)
            h_t_tilde = self.tanh(self.concat(torch.cat([
                decoder_output,
                context_vector
            ], dim=-1)))
            y_hat = self.generator(h_t_tilde)
            # |y_hat| = (batch_size, 1, output_size)
            y_hats.append(y_hat)

            if is_greedy:
                y = y_hat.argmax(dim=-1)
                # |y| = (batch_size, 1)
            else:
                # Take a random sampling based on the multinoulli distribution.
                y = torch.multinomial(y_hat.exp().view(batch_size, -1), 1)
                # |y| = (batch_size, 1)
            
            # Put PAD if the sample is done.
            y = y.masked_fill_(~is_decoding, dataloader.PAD)
            # Update is_decoding if there is EOS token.
            is_decoding = is_decoding * torch.ne(y, dataloader.EOS)
            # |is_decoding| = (batch_size, 1)
            indice.append(y)
        
        y_hats = torch.cat(y_hats, dim=1)
        indice = torch.cat(indice, dim=1)
        # |y_hats| = (batch_size, length, output_size)
        # |indice| = (batch_size, length)

        return y_hats, indice


    def batch_beam_search(
        self,
        src,
        beam_size=5,
        max_length=255,
        n_best=1,
        length_penalty=.2
    ):
        mask, x_length = None, None

        if isinstance(src, tuple):
            x, x_length = src
            mask = self.generate_mask(x, x_length)
        else:
            x = src
        batch_size = x.size(0)

        emb_src = self.emb_src(x)
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        # |h_src| = (batch_size, length, hidden_size)
        h_0_tgt = self.fast_merge_encoder_hiddens(h_0_tgt)
        # |h_0_tgt[0]| = (n_layers, batch_size, hidden_size)

        # Initialize 'SingleBeamSearchBoard' as many as beam_size
        boards = [SingleBeamSearchBoard(
            h_src.device,
            {
                'hidden_state': { # |h_0_tgt[0]| = (n_layers, batch_size, hidden_size)
                    'init_status': h_0_tgt[0][:, i, :].unsqueeze(1),
                    'batch_dim_index': 1
                },
                'cell_state': { # |h_0_tgt[1]| = (n_layers, batch_size, hidden_size)
                    'init_status': h_0_tgt[1][:, i, :].unsqueeze(1),
                    'batch_dim_index': 1
                },
                'h_t_1_tilde': { # |h_t_1_tilde| = (batch_size, 1, hidden_size)
                    'init_status': None,
                    'batch_dim_index': 0
                }
            },
            beam_size=beam_size,
            max_length=max_length
        ) for i in range(batch_size)]
        done_cnt = [board.is_done() for board in boards]

        length = 0
        # Run loop while sum of 'done_cnt' is smaller than batch_size,
        # or length is still smaller than max_length.
        while sum(done_cnt) < batch_size and length <= max_length:
            # current batch_size = sum(done_cnt) * beam_size

            # Initialize fabricated variables.
            fab_input, fab_hidden, fab_cell, fab_h_t_tilde = [], [], [], []
            fab_h_src, fab_mask = [], []

            # Build fabricated mini_batch in non-parallel way.
            # This may cause a bottle-neck.
            for i, board in enumerate(boards):
                # Batchify if the inference for the sample is still not finished.
                if board.is_done() == 0:
                    y_hat_i, prev_status = board.get_batch()

                    hidden_i    = prev_status['hidden_state']
                    cell_i      = prev_status['cell_state']
                    h_t_tilde_i = prev_status['h_t_1_tilde']

                    fab_input   += [y_hat_i]
                    fab_hidden  += [hidden_i]
                    fab_cell    += [cell_i]
                    fab_h_src   += [h_src[i, :, :]] * beam_size
                    fab_mask    += [mask[i, :]] * beam_size
                    if h_t_tilde_i is not None:
                        fab_h_t_tilde += [h_t_tilde_i]
                    else:
                        fab_h_t_tilde = None
            
            # Now, concatenate list of tensors.
            fab_input   = torch.cat(fab_input, dim=0)
            fab_hidden  = torch.cat(fab_hidden, dim=1)
            fab_cell    = torch.cat(fab_cell, dim=1)
            fab_h_src   = torch.stack(fab_h_src)
            fab_mask    = torch.stack(fab_mask)
            if fab_h_t_tilde is not None:
                fab_h_t_tilde = torch.cat(fab_h_t_tilde, dim=0)
            # |fab_input|     = (current_batch_size, 1)
            # |fab_hidden|    = (n_layers, current_batch_size, hidden_size)
            # |fab_cell|      = (n_layers, current_batch_size, hidden_size)
            # |fab_h_src|     = (current_batch_size, length, hidden_size)
            # |fab_mask|      = (current_batch_size, length)
            # |fab_h_t_tilde| = (current_batch_size, 1, hidden_size)

            emb_t = self.emb_tgt(fab_input)
            # |emb_t| = (current_batch_size, 1, embedding_dim)

            fab_decoder_output, (fab_hidden, fab_cell) = self.decoder(emb_t,
                                                                      fab_h_t_tilde,
                                                                      (fab_hidden, fab_cell))
            # |fab_decoder_output| = (current_batch_size, 1, hidden_size)
            context_vector = self.attn(fab_h_src, fab_decoder_output, fab_mask)
            # |context_vector| = (current_batch_size, 1, hidden_size)
            fab_h_t_tilde = self.tanh(self.concat(torch.cat([fab_decoder_output,
                                                             context_vector], dim=-1)))
            # |fab_h_t_tilde| = (current_batch_size, 1, hidden_size)
            y_hat = self.generator(fab_h_t_tilde)
            # |y_hat| = (current_batch_size, 1, output_size)

            # Separate the result for each sample.
            cnt = 0
            for board in boards:
                if board.is_done() == 0:
                    # Decide a range of each sample.
                    begin = cnt * beam_size
                    end = begin + beam_size

                    # Pick k-best results for each sample.
                    board.collect_result(
                        y_hat[begin:end],
                        {
                            'hidden_state': fab_hidden[:, begin:end, :],
                            'cell_state'  : fab_cell[:, begin:end, :],
                            'h_t_1_tilde' : fab_h_t_tilde[begin:end]
                        }
                    )
                    cnt += 1
            
            done_cnt = [board.is_done() for board in boards]
            length += 1
        
        # Pick n-best hypothesis.
        batch_sentences, batch_probs = [], []

        # Collect the results.
        for i, board in enumerate(boards):
            sentences, probs = board.get_n_best(n_best, length_penalty=length_penalty)

            batch_sentences += [sentences]
            batch_probs     += [probs]
        
        return batch_sentences, batch_probs


        


        