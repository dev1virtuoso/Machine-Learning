import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import math
import numpy as np
from scipy.spatial.distance import cdist

class AdversarialDiscriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, filter_sizes=[2, 3, 4], num_filters=100, dropout=0.5):
        super(AdversarialDiscriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 1)

    def get_embeddings(self, input_seqs):
        return self.embedding(input_seqs)

    def forward(self, input_features, is_embedding=False):
        if is_embedding:
            embedded = input_features.transpose(1, 2)
        else:
            embedded = self.embedding(input_features).transpose(1, 2)
        
        pooled_outputs = []
        for conv in self.convs:
            conved = F.relu(conv(embedded)) 
            pooled = F.max_pool1d(conved, conved.size(2)).squeeze(2) 
            pooled_outputs.append(pooled)
            
        cat = self.dropout(torch.cat(pooled_outputs, 1))
        return self.fc(cat)

class EvaluationMetrics:
    @staticmethod
    def compute_gradient_penalty(discriminator, real_embeddings, fake_embeddings):
        batch_size = real_embeddings.size(0)
        is_cuda = real_embeddings.is_cuda
        
        alpha = torch.rand(batch_size, 1, 1)
        alpha = alpha.expand(real_embeddings.size())
        if is_cuda: alpha = alpha.cuda()
            
        interpolates = alpha * real_embeddings.data + ((1 - alpha) * fake_embeddings.data)
        interpolates = Variable(interpolates, requires_grad=True)
        
        d_interpolates = discriminator(interpolates, is_embedding=True)
        
        fake = Variable(torch.ones(batch_size, 1), requires_grad=False)
        if is_cuda: fake = fake.cuda()
            
        gradients = autograd.grad(
            outputs=d_interpolates, inputs=interpolates,
            grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        
        gradients = gradients.contiguous().view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    @staticmethod
    def calculate_perplexity(log_probs, target_seqs, pad_token_id=0):
        vocab_size = log_probs.size(2)
        log_probs_flat = log_probs.contiguous().view(-1, vocab_size)
        targets_flat = target_seqs.contiguous().view(-1)
        
        loss_fn = nn.NLLLoss(size_average=False, ignore_index=pad_token_id)
        total_loss = loss_fn(log_probs_flat, targets_flat)
        
        num_tokens = int((targets_flat.data != pad_token_id).sum())
        if num_tokens == 0: return 0.0
        
        loss_val = float(total_loss.data[0]) if total_loss.data.dim() > 0 else float(total_loss.data)
        avg_loss = loss_val / num_tokens
        return math.exp(avg_loss)

    @staticmethod
    def _dtw_alignment(x, y):
        x_np, y_np = x.reshape(-1, 1), y.reshape(-1, 1)
        dist_mat = cdist(x_np, y_np, metric='euclidean')
        dtw_mat = np.zeros_like(dist_mat)
        dtw_mat[0, 0] = dist_mat[0, 0]
        
        for i in range(1, x_np.shape[0]): dtw_mat[i, 0] = dist_mat[i, 0] + dtw_mat[i-1, 0]
        for j in range(1, y_np.shape[0]): dtw_mat[0, j] = dist_mat[0, j] + dtw_mat[0, j-1]
            
        for i in range(1, x_np.shape[0]):
            for j in range(1, y_np.shape[0]):
                dtw_mat[i, j] = dist_mat[i, j] + min(dtw_mat[i-1, j], dtw_mat[i, j-1], dtw_mat[i-1, j-1])
                
        path_x, path_y = [x_np.shape[0]-1], [y_np.shape[0]-1]
        while path_x[-1] > 0 or path_y[-1] > 0:
            i, j = path_x[-1], path_y[-1]
            if i == 0: 
                path_x.append(0)
                path_y.append(j-1)
            elif j == 0: 
                path_x.append(i-1)
                path_y.append(0)
            else:
                steps = [dtw_mat[i-1, j-1], dtw_mat[i-1, j], dtw_mat[i, j-1]]
                min_step = np.argmin(steps)
                if min_step == 0: 
                    path_x.append(i-1); path_y.append(j-1)
                elif min_step == 1: 
                    path_x.append(i-1); path_y.append(j)
                else: 
                    path_x.append(i); path_y.append(j-1)
        return path_x[::-1], path_y[::-1]

    @staticmethod
    def calculate_prosody_correlation(pred_pitch, target_pitch, pad_mask=None):
        batch_size = pred_pitch.size(0)
        correlations = []
        
        pred_np = pred_pitch.data.cpu().numpy()
        target_np = target_pitch.data.cpu().numpy()
        mask_np = pad_mask.data.cpu().numpy() if pad_mask is not None else np.ones_like(pred_np)
        
        for i in range(batch_size):
            valid_len = int(np.sum(mask_np[i]))
            if valid_len < 2: continue
                
            p_valid = pred_np[i, :valid_len]
            t_valid = target_np[i, :valid_len]
            
            path_p, path_t = EvaluationMetrics._dtw_alignment(p_valid, t_valid)
            p_aligned = p_valid[path_p]
            t_aligned = t_valid[path_t]
            
            p_mean, t_mean = np.mean(p_aligned), np.mean(t_aligned)
            num = np.sum((p_aligned - p_mean) * (t_aligned - t_mean))
            den = np.sqrt(np.sum((p_aligned - p_mean)**2) * np.sum((t_aligned - t_mean)**2))
            
            if den > 1e-8: correlations.append(num / den)
            else: correlations.append(0.0)
                
        return np.mean(correlations) if correlations else 0.0