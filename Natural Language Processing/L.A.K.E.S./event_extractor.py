import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def log_sum_exp(vec):
    max_score, _ = torch.max(vec, 1)
    max_score_broadcast = max_score.view(-1, 1).expand_as(vec)
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 1))

class JointEventExtractor(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, tag_to_ix, num_role_tags, dropout=0.5, use_gpu=False):
        super(JointEventExtractor, self).__init__()
        self.use_gpu = use_gpu
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.num_role_tags = num_role_tags
        
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        
        self.trigger_head = nn.Linear(hidden_dim, self.tagset_size) 
        self.role_head = nn.Linear(hidden_dim * 2, num_role_tags)
        self.dropout_role = nn.Dropout(dropout)
        self.role_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix['<START>'], :] = -10000.0
        self.transitions.data[:, tag_to_ix['<STOP>']] = -10000.0

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

    def _score_sentence(self, feats, tags, mask):
        batch_size, seq_len, _ = feats.size()
        start_tags = Variable(tags.data.new(batch_size, 1).fill_(self.tag_to_ix['<START>']))
        pad_tags = torch.cat([start_tags, tags], 1)

        emit_scores = torch.gather(feats, 2, tags.unsqueeze(2)).squeeze(2) * mask
        trans_indices = pad_tags[:, 1:] * self.tagset_size + pad_tags[:, :-1]
        trans_scores = torch.gather(self.transitions.view(-1).unsqueeze(0).expand(batch_size, -1), 1, trans_indices) * mask
        
        mask_sum = mask.long().sum(1)
        last_indices = torch.clamp(mask_sum - 1, min=0).unsqueeze(1)
        last_tags = torch.gather(tags, 1, last_indices).squeeze(1)
        stop_trans = self.transitions[self.tag_to_ix['<STOP>'], last_tags]
        stop_mask = (mask_sum > 0).float()
        
        return torch.sum(emit_scores, 1) + torch.sum(trans_scores, 1) + (stop_trans * stop_mask)

    def _decode_triggers(self, feats, mask):
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
        
        bptrs_stacked = torch.stack(backpointers).data.cpu().numpy()
        best_tag_ids_np = best_tag_ids.data.cpu().numpy()
        mask_np = mask.data.cpu().numpy()
        
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
            
        return best_paths

    def forward(self, sentences, trigger_indices=None, entity_indices=None):
        batch_size, seq_len = sentences.size()
        embeds = self.dropout(self.embedding(sentences))
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        
        trigger_feats = self.trigger_head(lstm_out)
        role_logits = None
        
        if trigger_indices is not None and entity_indices is not None:
            t_idx = torch.clamp(trigger_indices.unsqueeze(1) if trigger_indices.dim() == 1 else trigger_indices, 0, seq_len - 1)
            e_idx = torch.clamp(entity_indices.unsqueeze(1) if entity_indices.dim() == 1 else entity_indices, 0, seq_len - 1)
            
            t_mask = ((t_idx >= 0) & (t_idx < seq_len)).float().unsqueeze(2)
            e_mask = ((e_idx >= 0) & (e_idx < seq_len)).float().unsqueeze(2)
            
            t_hidden = (torch.gather(lstm_out, 1, t_idx.unsqueeze(2).expand(batch_size, 1, lstm_out.size(2))) * t_mask).squeeze(1)
            e_hidden = (torch.gather(lstm_out, 1, e_idx.unsqueeze(2).expand(batch_size, 1, lstm_out.size(2))) * e_mask).squeeze(1)
            
            combined_features = self.dropout_role(torch.cat([t_hidden, e_hidden], dim=1))
            role_logits = self.role_head(combined_features)
            
        return trigger_feats, role_logits

    def predict(self, sentences, mask, entity_indices_batch):
        trigger_feats, _ = self.forward(sentences)
        trigger_paths = self._decode_triggers(trigger_feats, mask)
        
        batch_size = sentences.size(0)
        joint_predictions = []

        for b in range(batch_size):
            triggers = [i for i, tag in enumerate(trigger_paths[b]) if tag != self.tag_to_ix.get('O', 0)]
            entities = entity_indices_batch[b]
            
            b_results = {"triggers": trigger_paths[b], "roles": []}
            
            if triggers and entities:
                pairs_t = []
                pairs_e = []
                for t in triggers:
                    for e in entities:
                        pairs_t.append(t)
                        pairs_e.append(e)
                
                t_tensor = Variable(torch.LongTensor(pairs_t))
                e_tensor = Variable(torch.LongTensor(pairs_e))
                if self.use_gpu:
                    t_tensor, e_tensor = t_tensor.cuda(), e_tensor.cuda()
                
                single_sentence = sentences[b].unsqueeze(0).expand(len(pairs_t), -1)
                _, role_logits = self.forward(single_sentence, trigger_indices=t_tensor, entity_indices=e_tensor)
                
                role_preds = torch.max(role_logits, 1)[1].data.cpu().numpy()
                
                for idx, (t, e) in enumerate(zip(pairs_t, pairs_e)):
                    b_results["roles"].append((t, e, role_preds[idx]))
            
            joint_predictions.append(b_results)
            
        return joint_predictions

    def calculate_loss(self, sentences, mask, trigger_targets, trigger_indices=None, entity_indices=None, role_targets=None, alpha=1.0, beta=2.0, use_predicted_triggers=False):
        
        if use_predicted_triggers and entity_indices is not None:
            trigger_feats, _ = self.forward(sentences)
            pred_paths = self._decode_triggers(trigger_feats, mask)
            
            extracted_t = []
            for b in range(sentences.size(0)):
                trigs = [i for i, tag in enumerate(pred_paths[b]) if tag != self.tag_to_ix.get('O', 0)]
                extracted_t.append(trigs[0] if trigs else 0)
                
            trigger_indices = Variable(torch.LongTensor(extracted_t))
            if self.use_gpu: trigger_indices = trigger_indices.cuda()

        trigger_feats, role_logits = self.forward(sentences, trigger_indices, entity_indices)
        
        forward_score = self._forward_alg(trigger_feats, mask)
        gold_score = self._score_sentence(trigger_feats, trigger_targets, mask)
        trigger_loss = torch.mean(forward_score - gold_score)
        
        total_loss = trigger_loss * alpha
        if role_logits is not None and role_targets is not None:
            role_targets_flat = role_targets.view(-1)
            if role_targets_flat.size(0) > 0:
                role_loss = self.role_loss_fn(role_logits, role_targets_flat)
                total_loss += (role_loss * beta)
            
        return total_loss