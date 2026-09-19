import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random

class UtteranceEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, n_layers=1, dropout=0.2):
        super(UtteranceEncoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), bidirectional=True, batch_first=True)

    def forward(self, seq_inputs, lengths):
        embedded = self.embedding(seq_inputs)
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        seq_tensor = embedded[perm_idx]
        
        packed = nn.utils.rnn.pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy().tolist(), batch_first=True)
        _, hidden = self.gru(packed)
        
        _, unperm_idx = perm_idx.sort(0)
        hidden_unsorted = hidden[:, unperm_idx, :]
        return torch.cat((hidden_unsorted[-2], hidden_unsorted[-1]), 1) 

class ContextRNN(nn.Module):
    def __init__(self, utterance_dim, context_hidden_size, n_layers=1, dropout=0.2):
        super(ContextRNN, self).__init__()
        self.gru = nn.GRU(utterance_dim, context_hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), batch_first=True)

    def forward(self, utterance_vectors, hidden=None):
        return self.gru(utterance_vectors, hidden)

class HierarchicalLuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super(HierarchicalLuongAttention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)

    def forward(self, decoder_hidden, context_outputs, mask=None):
        score_input = self.attn(context_outputs)
        dec_transposed = decoder_hidden.transpose(0, 1)
        energies = torch.bmm(score_input, dec_transposed.transpose(1, 2)).squeeze(2)
        
        if mask is not None:
            energies.data.masked_fill_(mask.data, -float('inf'))
        return F.softmax(energies, dim=1).unsqueeze(1)

class HierarchicalStyleDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, tone_dim, num_tones, rhetoric_dim, n_layers=1, dropout=0.2):
        super(HierarchicalStyleDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.tone_embedding = nn.Embedding(num_tones, tone_dim)
        self.rhetoric_proj = nn.Linear(rhetoric_dim, 64) 
        self.dropout = nn.Dropout(dropout)
        
        self.gru = nn.GRU(embedding_dim + tone_dim + 64 + hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), batch_first=True)
        self.attention = HierarchicalLuongAttention(hidden_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_step, last_hidden, context_outputs, target_tone, target_rhetoric_vec, mask=None):
        word_emb = self.dropout(self.embedding(input_step))
        tone_emb = self.tone_embedding(target_tone).unsqueeze(1)
        rhetoric_emb = F.relu(self.rhetoric_proj(target_rhetoric_vec)).unsqueeze(1)
        
        attn_weights = self.attention(last_hidden[-1].unsqueeze(0), context_outputs, mask)
        context_vector = attn_weights.bmm(context_outputs) 
        
        gru_input = torch.cat((word_emb, tone_emb, rhetoric_emb, context_vector), 2)
        rnn_output, hidden = self.gru(gru_input, last_hidden)
        
        concat_output = F.tanh(self.concat(torch.cat((rnn_output.squeeze(1), context_vector.squeeze(1)), 1)))
        output = self.out(concat_output) 
        return output, hidden, attn_weights

class HarperHRED(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, tone_dim, num_tones, rhetoric_dim, n_layers=1):
        super(HarperHRED, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.encoder = UtteranceEncoder(vocab_size, embedding_dim, hidden_size, n_layers)
        self.context_rnn = ContextRNN(hidden_size * 2, hidden_size, n_layers)
        self.decoder = HierarchicalStyleDecoder(vocab_size, embedding_dim, hidden_size, tone_dim, num_tones, rhetoric_dim, n_layers)

    def forward(self, history_seqs, history_lengths, target_seqs, target_tones, target_rhetorics, global_step=0, max_steps=50000):
        batch_size, num_turns, _ = history_seqs.size()
        max_target_len = target_seqs.size(1)
        is_cuda = next(self.parameters()).is_cuda
        
        teacher_forcing_ratio = max(0.0, 1.0 - (float(global_step) / float(max_steps)))
        
        context_mask = Variable((history_lengths == 0).byte())
        if is_cuda: context_mask = context_mask.cuda()
        
        turn_vectors = []
        for turn in range(num_turns):
            lengths_t = history_lengths[:, turn]
            valid_mask = lengths_t > 0
            valid_idx = valid_mask.data.nonzero()
            
            turn_vec = Variable(torch.zeros(batch_size, self.hidden_size * 2))
            if is_cuda: turn_vec = turn_vec.cuda()
                
            if valid_idx.dim() > 0 and valid_idx.numel() > 0:
                valid_idx = valid_idx.squeeze(1)
                valid_idx_var = Variable(valid_idx)
                if is_cuda: valid_idx_var = valid_idx_var.cuda()
                
                valid_seqs = history_seqs.index_select(0, valid_idx_var)[:, turn, :]
                valid_lens = lengths_t.index_select(0, valid_idx_var)
                
                enc_out = self.encoder(valid_seqs, valid_lens)
                turn_vec.index_copy_(0, valid_idx_var, enc_out)
                
            turn_vectors.append(turn_vec.unsqueeze(1))
            
        context_inputs = torch.cat(turn_vectors, 1)
        context_outputs, context_hidden = self.context_rnn(context_inputs)
        
        if context_hidden.size(0) == self.decoder.n_layers:
            decoder_hidden = context_hidden.contiguous()
        else:
            decoder_hidden = context_hidden.repeat(self.decoder.n_layers, 1, 1).contiguous()
            
        decoder_input = Variable(torch.LongTensor([[1] * batch_size])).transpose(0, 1)
        if is_cuda: decoder_input = decoder_input.cuda()
            
        step_outputs = []
        for t in range(max_target_len):
            decoder_output, decoder_hidden, _ = self.decoder(
                decoder_input, decoder_hidden, context_outputs, target_tones, target_rhetorics, context_mask
            )
            step_outputs.append(F.log_softmax(decoder_output, dim=1))
            
            if random.random() < teacher_forcing_ratio:
                decoder_input = target_seqs[:, t].unsqueeze(1)
            else:
                decoder_input = decoder_output.max(1)[1].unsqueeze(1)
                
        return torch.stack(step_outputs, 1)