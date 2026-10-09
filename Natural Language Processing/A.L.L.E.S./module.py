import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import random

class Vocabulary:
    def __init__(self, name, min_freq=2):
        self.name = name
        self.word2index = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.word2count = {}
        self.index2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.n_words = 4
        self.min_freq = min_freq
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

    def __len__(self):
        return self.n_words

    def build_vocab_from_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                for word in line.strip().split():
                    self.word2count[word] = self.word2count.get(word, 0) + 1
        
        for word, count in self.word2count.items():
            if count >= self.min_freq and word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class StyleDiscriminator(nn.Module):
    def __init__(self, enc_hid_dim, num_styles, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(enc_hid_dim * 2, enc_hid_dim)
        self.fc2 = nn.Linear(enc_hid_dim, num_styles)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_hidden, alpha=1.0):
        reversed_hidden = GradientReversal.apply(encoder_hidden, alpha)
        x = self.dropout(torch.relu(self.fc1(reversed_hidden)))
        return self.fc2(x)

class MemoryBank(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, current_state, memory_tensor, history_lens=None):
        if memory_tensor is None or memory_tensor.size(1) == 0 or history_lens is None:
            return torch.zeros_like(current_state)
        
        batch_size, seq_len, hidden_dim = memory_tensor.size()
        state_expanded = current_state.unsqueeze(1).repeat(1, seq_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat((state_expanded, memory_tensor), dim=2)))
        attention_logits = energy.squeeze(2)
        
        mask = torch.arange(seq_len, device=memory_tensor.device).unsqueeze(0) < history_lens.unsqueeze(1)
        attention_logits = attention_logits.masked_fill(~mask, float('-inf'))
        
        attention = F.softmax(attention_logits, dim=1).unsqueeze(1)
        attention = torch.where(torch.isnan(attention), torch.zeros_like(attention), attention)
        
        context = torch.bmm(attention, memory_tensor).squeeze(1)
        return context

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.fc_hidden = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.fc_cell = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):
        embedded = self.dropout(self.embedding(src))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        
        combined_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        combined_cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)
        
        dec_hidden = torch.tanh(self.fc_hidden(combined_hidden))
        dec_cell = torch.tanh(self.fc_cell(combined_cell))
        
        return outputs, dec_hidden, dec_cell, combined_hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        src_len = encoder_outputs.shape[1]
        hidden_expanded = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden_expanded, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask == 0, float('-inf'))
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, num_langs, num_styles, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.attention = Attention(enc_hid_dim, dec_hid_dim)
        
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.lang_embedding = nn.Embedding(num_langs, emb_dim // 4)
        self.style_embedding = nn.Embedding(num_styles, emb_dim // 4)
        
        rnn_input_dim = emb_dim + (emb_dim // 4 * 2) + (enc_hid_dim * 2) + dec_hid_dim
        self.rnn = nn.LSTM(rnn_input_dim, dec_hid_dim, num_layers=1, batch_first=True)
        self.fc_out = nn.Linear(dec_hid_dim + (enc_hid_dim * 2) + emb_dim + (emb_dim // 4 * 2), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs, mask, lang_token, style_token, mem_context):
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))
        
        lang_embedded = self.lang_embedding(lang_token).unsqueeze(1)
        style_embedded = self.style_embedding(style_token).unsqueeze(1)
        
        a = self.attention(hidden, encoder_outputs, mask).unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        
        mem_expanded = mem_context.unsqueeze(1)
        rnn_input = torch.cat((embedded, lang_embedded, style_embedded, weighted, mem_expanded), dim=2)
        
        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        
        embedded = embedded.squeeze(1)
        lang_embedded = lang_embedded.squeeze(1)
        style_embedded = style_embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded, lang_embedded, style_embedded), dim=1))
        return prediction, hidden.squeeze(0), cell.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, sos_idx=1, eos_idx=2):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.memory = MemoryBank(decoder.rnn.hidden_size)

    def create_mask(self, src):
        return (src != self.src_pad_idx)

    def forward(self, src, src_lengths, trg, lang_token, style_token, memory_tensor=None, history_lens=None, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)
        encoder_outputs, hidden, cell, raw_enc_hidden = self.encoder(src, src_lengths)
        mask = self.create_mask(src)
        
        mem_context = self.memory(hidden, memory_tensor, history_lens)
        
        input_token = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs, mask, lang_token, style_token, mem_context)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t] if teacher_force else top1
            
        return outputs, raw_enc_hidden, hidden

    def greedy_decode(self, src, src_lengths, lang_token, style_token, memory_tensor=None, history_lens=None, max_len=50):
        batch_size = src.shape[0]
        encoder_outputs, hidden, cell, raw_enc_hidden = self.encoder(src, src_lengths)
        mask = self.create_mask(src)
        mem_context = self.memory(hidden, memory_tensor, history_lens)
        
        input_token = (torch.ones(batch_size).long().to(src.device)) * self.sos_idx
        decoded_batch = torch.zeros(batch_size, max_len).long().to(src.device)
        
        for t in range(max_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs, mask, lang_token, style_token, mem_context)
            top1 = output.argmax(1)
            decoded_batch[:, t] = top1
            input_token = top1
            
        return decoded_batch, hidden