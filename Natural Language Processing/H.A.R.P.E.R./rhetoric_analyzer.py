import spacy
import torch
import torch.nn as nn
from torch.autograd import Variable

class RhetoricBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(RhetoricBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        seq_tensor = embedded[perm_idx]
        
        packed = nn.utils.rnn.pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy().tolist(), batch_first=True)
        _, (hidden, _) = self.lstm(packed)
        
        _, unperm_idx = perm_idx.sort(0)
        hidden_unsorted = hidden[:, unperm_idx, :]
        
        hidden_cat = torch.cat((hidden_unsorted[-2], hidden_unsorted[-1]), 1)
        return self.fc(hidden_cat), hidden_cat

class RhetoricAnalyzer:
    def __init__(self, spacy_model='en_core_web_sm', ml_model_path=None, vocab_dict=None):
        self.nlp = spacy.load(spacy_model)
        self.text_vocab = vocab_dict if vocab_dict is not None else {}
        self.is_cuda = torch.cuda.is_available()
        self.ml_model = None
        self.rhetoric_dim = 256 
        
        if ml_model_path and vocab_dict:
            self.ml_model = RhetoricBiLSTM(len(vocab_dict), 300, 128, 5) 
            self.ml_model.load_state_dict(torch.load(ml_model_path))
            if self.is_cuda: self.ml_model.cuda()
            self.ml_model.eval()

    def analyze_text(self, text):
        doc = self.nlp(text)
        features = {"anaphora": 0, "chiasmus": 0, "rhetorical_question": 0, "metaphor": 0}
        
        if self.ml_model and len(self.text_vocab) > 0:
            continuous_embedding = self._run_ml_model(doc)
        else:
            val = torch.zeros(1, self.rhetoric_dim)
            continuous_embedding = Variable(val.cuda() if self.is_cuda else val, requires_grad=False)
            
        if len(doc) == 0: 
            return features, continuous_embedding

        if self._detect_anaphora(doc): features["anaphora"] = 1
        if self._detect_chiasmus(doc): features["chiasmus"] = 1
        if self._detect_rhetorical_question(doc): features["rhetorical_question"] = 1
        if self._detect_metaphor(doc): features["metaphor"] = 1
            
        return features, continuous_embedding

    def _detect_anaphora(self, doc):
        sentences = [sent for sent in doc.sents if len(sent) > 2]
        if len(sentences) < 2: return False
        for i in range(len(sentences) - 1):
            s1_head = [t for t in sentences[i] if not t.is_punct and t.is_alpha]
            s2_head = [t for t in sentences[i+1] if not t.is_punct and t.is_alpha]
            if s1_head and s2_head and s1_head[0].lower_ == s2_head[0].lower_:
                return True
        return False

    def _detect_chiasmus(self, doc):
        valid_pos = ['NOUN', 'VERB', 'ADJ', 'PRON']
        tags = [(token.pos_, token.dep_, token.lemma_) for token in doc if not token.is_punct]
        for i in range(len(tags) - 3):
            t1, t2, t3, t4 = tags[i], tags[i+1], tags[i+2], tags[i+3]
            if (t1[0] == t4[0] and t2[0] == t3[0] and t1[0] != t2[0]) or \
               (t1[2] == t4[2] and t2[2] == t3[2]):
                if t1[0] in valid_pos and t2[0] in valid_pos:
                    return True
        return False

    def _detect_rhetorical_question(self, doc):
        if len(doc) == 0 or doc[-1].text != '?': return False
        interrogatives = ['who', 'what', 'when', 'where', 'why', 'how']
        if doc[0].lower_ in interrogatives and any(t.dep_ == 'neg' for t in doc):
            return True
        if len(doc) > 2 and doc[0].pos_ in ['AUX', 'VERB'] and doc[1].dep_ == 'neg':
            return True
        return False

    def _detect_metaphor(self, doc):
        for token in doc:
            if token.lower_ in ['like', 'as'] and token.dep_ == 'prep':
                head = token.head
                if head.pos_ in ['VERB', 'NOUN'] and list(token.children):
                    return True
        return False

    def _run_ml_model(self, doc):
        if len(doc) == 0:
            val = torch.zeros(1, self.rhetoric_dim)
            return Variable(val.cuda() if self.is_cuda else val, requires_grad=False)
        tokens = [self.text_vocab.get(token.text.lower(), 0) for token in doc]
        token_tensor = Variable(torch.LongTensor([tokens]))
        length_tensor = torch.LongTensor([len(tokens)])
        if self.is_cuda: 
            token_tensor = token_tensor.cuda()
            length_tensor = length_tensor.cuda()
            
        _, continuous_emb = self.ml_model(token_tensor, length_tensor)
        return continuous_emb