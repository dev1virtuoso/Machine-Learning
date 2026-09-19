import numpy as np
import torch
from torch.autograd import Variable
from collections import defaultdict

def _build_inverted_index(knowledge_base):
    index = defaultdict(list)
    kb_lower = {str(k).lower(): v for k, v in knowledge_base.items()}
    
    sorted_keys = sorted(kb_lower.keys(), key=lambda x: len(x.split()), reverse=True)
    
    for key in sorted_keys:
        tokens = frozenset(key.split())
        index[tokens].append((key, kb_lower[key]))
    return index, kb_lower

def align_distant_supervision(tokens, knowledge_base, max_ngram=4, jaccard_threshold=0.75):
    if not tokens or not knowledge_base:
        return []

    labels = ["O"] * len(tokens)
    occupied = [False] * len(tokens)
    text_lower = [t.lower() for t in tokens]
    kb_index, kb_lower = _build_inverted_index(knowledge_base)
    
    for n in range(max_ngram, 0, -1):
        for i in range(len(text_lower) - n + 1):
            if any(occupied[i:i+n]):
                continue
                
            ngram_str = " ".join(text_lower[i:i+n])
            relation = None
            
            if ngram_str in kb_lower:
                relation = kb_lower[ngram_str]
            else:
                ngram_tokens = set(text_lower[i:i+n])
                best_score = 0.0
                
                for kb_tokens, items in kb_index.items():
                    intersection = len(ngram_tokens & kb_tokens)
                    union = len(ngram_tokens | kb_tokens)
                    score = intersection / float(union)
                    
                    if score > jaccard_threshold and score > best_score:
                        best_score = score
                        relation = items[0][1] 
                        
            if relation:        
                labels[i] = f"B-{relation}"
                occupied[i] = True
                for j in range(1, n):
                    labels[i+j] = f"I-{relation}"
                    occupied[i+j] = True
                    
    return labels

def get_position_ids(seq_lens, head_indices, tail_indices, max_len=100, use_gpu=False):
    if not head_indices or not tail_indices:
        raise ValueError("Entity indices cannot be empty.")
        
    batch_size = len(head_indices)
    max_seq_len = int(max(seq_lens))
    
    idx_grid = np.arange(max_seq_len)[None, :]
    h_idx = np.array(head_indices)[:, None]
    t_idx = np.array(tail_indices)[:, None]
    
    pos1 = idx_grid - h_idx + max_len
    pos2 = idx_grid - t_idx + max_len
    
    pos1 = np.clip(pos1, 0, (2 * max_len) - 1).astype(np.int64)
    pos2 = np.clip(pos2, 0, (2 * max_len) - 1).astype(np.int64)
    
    seq_lens_arr = np.array(seq_lens)[:, None]
    mask = idx_grid < seq_lens_arr
    
    pos1 = np.where(mask, pos1, max_len)
    pos2 = np.where(mask, pos2, max_len)
    
    p1_var = Variable(torch.LongTensor(pos1))
    p2_var = Variable(torch.LongTensor(pos2))
    
    if use_gpu:
        p1_var = p1_var.cuda()
        p2_var = p2_var.cuda()
        
    return p1_var, p2_var