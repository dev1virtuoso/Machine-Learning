import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, n_layers=2, dropout=0.2):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            embedding_dim, hidden_size, n_layers, 
            dropout=(0 if n_layers == 1 else dropout), 
            bidirectional=True, batch_first=True
        )

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)
        seq_lengths, perm_idx = input_lengths.sort(0, descending=True)
        seq_tensor = embedded[perm_idx]
        
        packed = nn.utils.rnn.pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy().tolist(), batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        _, unperm_idx = perm_idx.sort(0)
        outputs = outputs[unperm_idx]
        hidden = hidden[:, unperm_idx, :]
        
        outputs = outputs.contiguous().view(outputs.size(0), outputs.size(1), 2, self.hidden_size).sum(2)
        return outputs, hidden

class LuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super(LuongAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, encoder_outputs, mask=None):
        energy = self.attn(encoder_outputs)
        hidden_expanded = hidden.transpose(0, 1)
        attn_energies = torch.bmm(energy, hidden_expanded.transpose(1, 2)).squeeze(2)
        
        if mask is not None:
            attn_energies.data.masked_fill_(mask.data, -float('inf'))
            
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class ConditionalDecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, tone_dim, num_tones, rhetoric_dim, n_layers=2, dropout=0.2):
        super(ConditionalDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.tone_embedding = nn.Embedding(num_tones, tone_dim)
        self.rhetoric_proj = nn.Linear(rhetoric_dim, 64) 
        self.embedding_dropout = nn.Dropout(dropout)
        
        self.gru = nn.GRU(embedding_dim + tone_dim + 64, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), batch_first=True)
        self.attention = LuongAttention(hidden_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_step, last_hidden, encoder_outputs, target_tone, target_rhetoric_vec, mask=None):
        word_embedded = self.embedding_dropout(self.embedding(input_step)) 
        tone_embedded = self.tone_embedding(target_tone).unsqueeze(1)      
        rhetoric_embedded = F.relu(self.rhetoric_proj(target_rhetoric_vec)).unsqueeze(1) 
        
        rnn_input = torch.cat((word_embedded, tone_embedded, rhetoric_embedded), 2) 
        
        rnn_output, hidden = self.gru(rnn_input, last_hidden)
        attn_weights = self.attention(hidden[-1].unsqueeze(0), encoder_outputs, mask)
        context = attn_weights.bmm(encoder_outputs)
        
        concat_input = torch.cat((rnn_output.squeeze(1), context.squeeze(1)), 1)
        concat_output = F.tanh(self.concat(concat_input)) 
        
        output = self.out(concat_output) 
        return output, hidden, attn_weights

class HarperSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(HarperSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seqs, input_lengths, target_seqs, target_tones, target_rhetoric_vecs, global_step=0, max_steps=50000):
        batch_size = input_seqs.size(0)
        max_target_len = target_seqs.size(1)
        is_cuda = next(self.parameters()).is_cuda
        
        teacher_forcing_ratio = max(0.0, 1.0 - (float(global_step) / float(max_steps)))
        
        mask = Variable((input_seqs.data == 0).byte()) 
        if is_cuda: mask = mask.cuda()
            
        encoder_outputs, encoder_hidden = self.encoder(input_seqs, input_lengths)
        
        decoder_hidden = encoder_hidden.contiguous().view(
            self.encoder.n_layers, 2, batch_size, self.encoder.hidden_size
        ).sum(1).expand(self.decoder.n_layers, -1, -1).contiguous()
        
        decoder_input = Variable(torch.LongTensor([[1] * batch_size])).transpose(0, 1)
        if is_cuda: decoder_input = decoder_input.cuda()
            
        step_outputs = []
            
        for t in range(max_target_len):
            decoder_output, decoder_hidden, _ = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, target_tones, target_rhetoric_vecs, mask
            )
            step_outputs.append(F.log_softmax(decoder_output, dim=1))
            
            if random.random() < teacher_forcing_ratio:
                decoder_input = target_seqs[:, t].unsqueeze(1)
            else:
                decoder_input = decoder_output.max(1)[1].unsqueeze(1)
                
        outputs = torch.stack(step_outputs, 1) 
        return outputs