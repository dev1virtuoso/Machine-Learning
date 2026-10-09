import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

def log_sum_exp(vec):
    max_score, _ = torch.max(vec, 1)
    max_score_broadcast = max_score.view(-1, 1).expand_as(vec)
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 1))

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, dropout=0.5, use_gpu=False):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.use_gpu = use_gpu

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix['<START>'], :] = -10000.0
        self.transitions.data[:, tag_to_ix['<STOP>']] = -10000.0

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        h0 = Variable(weight.new(2, batch_size, self.hidden_dim // 2).normal_(0, 0.1), requires_grad=False)
        c0 = Variable(weight.new(2, batch_size, self.hidden_dim // 2).normal_(0, 0.1), requires_grad=False)
        return (h0, c0)

    def _get_lstm_features(self, sentences, seq_lens):
        hidden = self.init_hidden(sentences.size(0))
        embeds = self.dropout(self.word_embeds(sentences))
        
        lengths_list = seq_lens.data.cpu().numpy().tolist() if isinstance(seq_lens, Variable) else seq_lens

        sorted_lens, sort_idx = torch.sort(torch.LongTensor(lengths_list), descending=True)
        _, unsort_idx = torch.sort(sort_idx)
        
        sort_idx_var = Variable(sort_idx)
        unsort_idx_var = Variable(unsort_idx)
        if self.use_gpu:
            sort_idx_var = sort_idx_var.cuda()
            unsort_idx_var = unsort_idx_var.cuda()
            
        embeds = embeds.index_select(0, sort_idx_var)
        h0 = hidden[0].index_select(1, sort_idx_var)
        c0 = hidden[1].index_select(1, sort_idx_var)
        
        packed_embeds = pack_padded_sequence(embeds, sorted_lens.tolist(), batch_first=True)
        
        lstm_out, _ = self.lstm(packed_embeds, (h0, c0))
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        lstm_out = lstm_out.index_select(0, unsort_idx_var)
        
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags, mask):
        batch_size, seq_len, _ = feats.size()
        
        start_tags = Variable(tags.data.new(batch_size, 1).fill_(self.tag_to_ix['<START>']))
        pad_tags = torch.cat([start_tags, tags], 1)

        emit_scores = torch.gather(feats, 2, tags.unsqueeze(2)).squeeze(2)
        emit_scores = emit_scores * mask
        
        trans_flat = self.transitions.view(-1)
        trans_indices = pad_tags[:, 1:] * self.tagset_size + pad_tags[:, :-1]
        trans_scores = torch.gather(trans_flat.unsqueeze(0).expand(batch_size, -1), 1, trans_indices)
        trans_scores = trans_scores * mask
        
        mask_sum = mask.long().sum(1)
        last_indices = torch.clamp(mask_sum - 1, min=0).unsqueeze(1)
        last_tags = torch.gather(tags, 1, last_indices).squeeze(1)
        
        stop_trans = self.transitions[self.tag_to_ix['<STOP>'], last_tags]
        stop_mask = (mask_sum > 0).float()
        
        score = torch.sum(emit_scores, 1) + torch.sum(trans_scores, 1) + (stop_trans * stop_mask)
        return score

    def _forward_alg(self, feats, mask):
        batch_size, seq_len, _ = feats.size()
        
        init_alphas = feats.data.new(batch_size, self.tagset_size).fill_(-10000.)
        init_alphas[:, self.tag_to_ix['<START>']] = 0.
        forward_var = Variable(init_alphas)

        trans = self.transitions.unsqueeze(0)
        for t in range(seq_len):
            emit_score = feats[:, t, :].unsqueeze(2)
            mask_t = mask[:, t].unsqueeze(1)
            next_tag_var = forward_var.unsqueeze(1) + trans + emit_score
            
            max_score, _ = torch.max(next_tag_var, 2)
            max_score_broadcast = max_score.unsqueeze(2).expand_as(next_tag_var)
            new_forward_var = max_score + torch.log(torch.sum(torch.exp(next_tag_var - max_score_broadcast), 2))
            forward_var = mask_t * new_forward_var + (1.0 - mask_t) * forward_var

        terminal_var = forward_var + self.transitions[self.tag_to_ix['<STOP>']].unsqueeze(0)
        return log_sum_exp(terminal_var)

    def decode(self, feats, mask):
        batch_size, seq_len, _ = feats.size()
        
        init_vvars = feats.data.new(batch_size, self.tagset_size).fill_(-10000.)
        init_vvars[:, self.tag_to_ix['<START>']] = 0.
        forward_var = Variable(init_vvars)
            
        backpointers = []
        trans = self.transitions.unsqueeze(0)
        
        for t in range(seq_len):
            mask_t = mask[:, t].unsqueeze(1)
            feat = feats[:, t, :]
            
            next_tag_var = forward_var.unsqueeze(1) + trans
            best_tag_scores, best_tag_ids = torch.max(next_tag_var, 2)
            
            new_forward_var = best_tag_scores + feat
            forward_var = mask_t * new_forward_var + (1.0 - mask_t) * forward_var
            backpointers.append(best_tag_ids)
            
        terminal_var = forward_var + self.transitions[self.tag_to_ix['<STOP>']].unsqueeze(0)
        best_tag_scores, best_tag_ids = torch.max(terminal_var, 1)
        
        best_tag_ids_np = best_tag_ids.data.cpu().numpy()
        mask_np = mask.data.cpu().numpy()
        bptrs_stacked = torch.stack(backpointers).data.cpu().numpy()
        
        best_paths = []
        for b in range(batch_size):
            length = int(mask_np[b].sum())
            if length == 0:
                best_paths.append([])
                continue
            best_tag = best_tag_ids_np[b]
            path = [best_tag]
            for t in reversed(range(length)):
                best_tag = bptrs_stacked[t, b, best_tag]
                path.append(best_tag)
            path.pop() 
            path.reverse()
            best_paths.append(path)
            
        return best_tag_scores, best_paths

    def neg_log_likelihood(self, sentences, seq_lens, tags, mask):
        if torch.sum(mask).data[0] == 0:
            zero_loss = Variable(sentences.data.new([0.0]).float(), requires_grad=True)
            if self.use_gpu:
                zero_loss = zero_loss.cuda()
            return zero_loss

        feats = self._get_lstm_features(sentences, seq_lens)
        forward_score = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(feats, tags, mask)
        return torch.mean(forward_score - gold_score)

    def forward(self, sentence, seq_lens, mask=None):
        if mask is None:
            mask = Variable((sentence.data != 0).float())
        lstm_feats = self._get_lstm_features(sentence, seq_lens)
        return self.decode(lstm_feats, mask)