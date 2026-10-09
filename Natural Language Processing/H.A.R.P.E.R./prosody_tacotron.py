import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import librosa

class PreNet(nn.Module):
    def __init__(self, in_dim, sizes=[256, 128]):
        super(PreNet, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(in_dim if i == 0 else sizes[i-1], sizes[i]) for i in range(len(sizes))])
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        for layer in self.layers:
            x = self.dropout(F.relu(layer(x)))
        return x

class Highway(nn.Module):
    def __init__(self, size):
        super(Highway, self).__init__()
        self.H = nn.Linear(size, size)
        self.T = nn.Linear(size, size)

    def forward(self, x):
        h = F.relu(self.H(x))
        t = F.sigmoid(self.T(x))
        return h * t + x * (1.0 - t)

class CBHG(nn.Module):
    def __init__(self, in_dim, K=16, projection_dims=[128, 128], gru_dim=128):
        super(CBHG, self).__init__()
        self.in_dim = in_dim
        self.K = K
        self.conv_bank = nn.ModuleList([nn.Conv1d(in_dim, in_dim, kernel_size=k) for k in range(1, K + 1)])
        self.conv_projection_1 = nn.Conv1d(in_dim * K, projection_dims[0], kernel_size=3, padding=1)
        self.conv_projection_2 = nn.Conv1d(projection_dims[0], in_dim, kernel_size=3, padding=1)
        self.batch_norm_1 = nn.BatchNorm1d(projection_dims[0])
        self.batch_norm_2 = nn.BatchNorm1d(in_dim)
        self.highway_layers = nn.ModuleList([Highway(in_dim) for _ in range(4)])
        self.gru = nn.GRU(in_dim, gru_dim, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x):
        x_trans = x.transpose(1, 2)
        bank_outputs = []
        for k, conv in enumerate(self.conv_bank):
            pad_total = k 
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            padded_x = F.pad(x_trans, (pad_left, pad_right))
            bank_outputs.append(conv(padded_x))
            
        joined_bank = torch.cat(bank_outputs, 1)
        pooled = F.max_pool1d(joined_bank, kernel_size=2, stride=1, padding=1)
        pooled = pooled[:, :, :x_trans.size(2)] 
        
        proj = F.relu(self.batch_norm_1(self.conv_projection_1(pooled)))
        proj = self.batch_norm_2(self.conv_projection_2(proj))
        
        highway_input = (proj + x_trans).transpose(1, 2)
        for layer in self.highway_layers: highway_input = layer(highway_input)
            
        outputs, _ = self.gru(highway_input)
        return outputs

class LocationSensitiveAttention(nn.Module):
    def __init__(self, query_dim, memory_dim, attention_dim, attention_location_n_filters=32, attention_location_kernel_size=31):
        super(LocationSensitiveAttention, self).__init__()
        self.query_layer = nn.Linear(query_dim, attention_dim, bias=False)
        self.memory_layer = nn.Linear(memory_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.location_layer = nn.Conv1d(1, attention_location_n_filters, kernel_size=attention_location_kernel_size, padding=(attention_location_kernel_size - 1) // 2)
        self.location_weight = nn.Linear(attention_location_n_filters, attention_dim, bias=False)

    def forward(self, query, memory, cumulative_weights):
        query_rep = self.query_layer(query).unsqueeze(1)
        memory_rep = self.memory_layer(memory)
        loc_features = self.location_layer(cumulative_weights.unsqueeze(1)).transpose(1, 2)
        loc_rep = self.location_weight(loc_features)
        energies = self.v(F.tanh(query_rep + memory_rep + loc_rep)).squeeze(2)
        weights = F.softmax(energies, dim=1)
        context = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)
        next_cumulative = cumulative_weights + weights
        return context, weights, next_cumulative

class PostNet(nn.Module):
    def __init__(self, mel_dim, linear_dim=1025):
        super(PostNet, self).__init__()
        self.cbhg = CBHG(mel_dim)
        self.linear_project = nn.Linear(256, linear_dim)
        
    def forward(self, mel_outputs):
        cbhg_out = self.cbhg(mel_outputs)
        linear_spec = self.linear_project(cbhg_out)
        return linear_spec

class GriffinLimVocoder:
    @staticmethod
    def synthesize(linear_spectrogram, n_iter=50, n_fft=2048, hop_length=256, win_length=1024):
        S = linear_spectrogram.T
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex128) * angles
        
        for i in range(n_iter):
            y = librosa.istft(S_complex, hop_length=hop_length, win_length=win_length)
            stft_matrix = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            angles = np.exp(1j * np.angle(stft_matrix))
            S_complex = np.abs(S).astype(np.complex128) * angles
            
        return librosa.istft(S_complex, hop_length=hop_length, win_length=win_length)

class ProsodyAcousticDecoder(nn.Module):
    def __init__(self, mel_dim, encoder_hidden_dim, style_dim, num_styles, prosody_feature_dim=3):
        super(ProsodyAcousticDecoder, self).__init__()
        self.mel_dim = mel_dim
        self.style_dim = style_dim
        self.style_embedding = nn.Embedding(num_styles, style_dim)
        self.prosody_dense = nn.Linear(prosody_feature_dim, 64)
        
        fused_memory_dim = encoder_hidden_dim + style_dim + 64
        self.attention = LocationSensitiveAttention(query_dim=256, memory_dim=fused_memory_dim, attention_dim=128)
        
        decoder_input_dim = mel_dim + fused_memory_dim + style_dim + 64
        self.decoder_rnn = nn.GRU(decoder_input_dim, 256, num_layers=2, batch_first=True)
        self.mel_projector = nn.Linear(256, mel_dim)
        self.stop_projector = nn.Linear(256, 1) 
        self.postnet = PostNet(mel_dim)

    def forward(self, encoder_memory, target_style, pitch_energy_duration, max_mel_steps=500, stop_threshold=0.8):
        batch_size = encoder_memory.size(0)
        seq_len = encoder_memory.size(1)
        is_cuda = next(self.parameters()).is_cuda
        
        current_frame = Variable(torch.zeros(batch_size, 1, self.mel_dim))
        decoder_hidden = Variable(torch.zeros(2, batch_size, 256))
        cumulative_weights = Variable(torch.zeros(batch_size, seq_len))
        
        if is_cuda:
            current_frame, decoder_hidden, cumulative_weights = current_frame.cuda(), decoder_hidden.cuda(), cumulative_weights.cuda()
            
        style_vec = self.style_embedding(target_style).unsqueeze(1) 
        prosody_vec = F.relu(self.prosody_dense(pitch_energy_duration)).unsqueeze(1) 
        
        style_expanded = style_vec.expand(-1, seq_len, -1)
        prosody_expanded = prosody_vec.expand(-1, seq_len, -1)
        fused_encoder_memory = torch.cat((encoder_memory, style_expanded, prosody_expanded), 2)
        
        mel_outputs, stop_outputs = [], []
        
        for step in range(max_mel_steps):
            query = decoder_hidden[-1]
            context_vector, _, cumulative_weights = self.attention(query, fused_encoder_memory, cumulative_weights)
            context_vector = context_vector.unsqueeze(1)
            
            decoder_step_input = torch.cat((current_frame, context_vector, style_vec, prosody_vec), 2)
            rnn_out, decoder_hidden = self.decoder_rnn(decoder_step_input, decoder_hidden)
            
            current_frame = self.mel_projector(rnn_out)
            stop_token = F.sigmoid(self.stop_projector(rnn_out))
            
            mel_outputs.append(current_frame)
            stop_outputs.append(stop_token)
            
            if (stop_token.data > stop_threshold).all() and step > 20:
                break
            
        mel_seq = torch.cat(mel_outputs, 1)
        linear_seq = self.postnet(mel_seq) 
        return mel_seq, linear_seq, torch.cat(stop_outputs, 1)