import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class PositionAwareRelationExtractor(nn.Module):
    def __init__(self, vocab_size, emb_dim, pos_dim, hidden_dim, num_relations, max_len=100, dropout=0.5):
        super(PositionAwareRelationExtractor, self).__init__()
        self.max_len = max_len
        self.word_embeds = nn.Embedding(vocab_size, emb_dim)
        self.pos_embeds = nn.Embedding((2 * max_len), pos_dim) 
        
        self.dropout_emb = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim + 2 * pos_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.dropout_lstm = nn.Dropout(dropout)
        
        self.attn_layer = nn.Linear(hidden_dim, 1)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_attn_out = nn.Dropout(dropout)
        
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_relations)

    def forward(self, words, pos1, pos2, mask, seq_lens):
        assert words.size(0) == pos1.size(0) == mask.size(0), "Batch size mismatch across input tensors."
        
        lengths_list = seq_lens.data.cpu().numpy().tolist() if isinstance(seq_lens, Variable) else seq_lens
        sorted_lens, sort_idx = torch.sort(torch.LongTensor(lengths_list), descending=True)
        _, unsort_idx = torch.sort(sort_idx)
        
        sort_idx = Variable(sort_idx)
        unsort_idx = Variable(unsort_idx)
        
        if words.is_cuda:
            sort_idx = sort_idx.cuda()
            unsort_idx = unsort_idx.cuda()

        words = words.index_select(0, sort_idx)
        pos1 = pos1.index_select(0, sort_idx)
        pos2 = pos2.index_select(0, sort_idx)
        mask = mask.index_select(0, sort_idx)
        
        w_emb = self.word_embeds(words)
        p1_emb = self.pos_embeds(pos1)
        p2_emb = self.pos_embeds(pos2)
        
        inputs = self.dropout_emb(torch.cat([w_emb, p1_emb, p2_emb], 2))
        
        packed_inputs = pack_padded_sequence(inputs, sorted_lens.tolist(), batch_first=True)
        lstm_out_packed, _ = self.lstm(packed_inputs)
        lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)
        
        lstm_out = self.dropout_lstm(lstm_out)
        attn_logits = self.attn_layer(lstm_out).squeeze(2)
        
        mask_var = (mask == 0)
        attn_logits = attn_logits.masked_fill(mask_var, -1e9)
        attn_weights = F.softmax(attn_logits, dim=1).unsqueeze(2)
        
        context = torch.bmm(attn_weights.transpose(1, 2), lstm_out).squeeze(1)
        
        projected = self.projection(context)
        if projected.size(0) > 1:
            projected = self.batch_norm(projected)
            
        out = torch.tanh(projected)
        out = self.dropout_attn_out(out)
        logits = self.classifier(out)
        
        return logits.index_select(0, unsort_idx)