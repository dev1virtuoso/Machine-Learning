import os
import sys
import json
import argparse
import logging
from collections import defaultdict

import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from bilstm_crf import BiLSTM_CRF
from event_extractor import JointEventExtractor
from relation_extractor import PositionAwareRelationExtractor
from linguistic_pipeline import DeepLinguisticAnalyzer
from data_utils import align_distant_supervision, get_position_ids

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.tag2idx = {"<PAD>": 0, "O": 1, "<START>": 2, "<STOP>": 3}
        self.rel2idx = {"<PAD>": 0, "NA": 1}
        self.idx2word, self.idx2tag, self.idx2rel = {}, {}, {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.word2idx)
            
    def add_tag(self, tag):
        if tag not in self.tag2idx:
            self.tag2idx[tag] = len(self.tag2idx)
            
    def add_relation(self, rel):
        if rel not in self.rel2idx:
            self.rel2idx[rel] = len(self.rel2idx)

    def compile(self):
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}
        self.idx2rel = {v: k for k, v in self.rel2idx.items()}

    def get_word_idx(self, word):
        return self.word2idx.get(word, self.word2idx["<UNK>"])

def load_data(filepath, vocab, is_train=True):
    dataset = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
        
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            tokens = data.get("tokens", [])
            tags = data.get("ner_tags", ["O"] * len(tokens))
            relations = data.get("relations", [])
            
            if is_train:
                for token in tokens: vocab.add_word(token)
                for tag in tags: vocab.add_tag(tag)
                for rel in relations: vocab.add_relation(rel.get("type", "NA"))
                
            dataset.append({
                "tokens": tokens,
                "tags": tags,
                "relations": relations
            })
    return dataset

def pad_sequence(seq, max_len, pad_value=0):
    return seq + [pad_value] * (max_len - len(seq))

def prepare_batch(batch_data, vocab, use_gpu):
    seq_lens = [len(item["tokens"]) for item in batch_data]
    max_len = max(seq_lens)
    
    sentences_idx = [[vocab.get_word_idx(w) for w in item["tokens"]] for item in batch_data]
    tags_idx = [[vocab.tag2idx.get(t, 0) for t in item["tags"]] for item in batch_data]
    
    sentences_pad = [pad_sequence(s, max_len, vocab.word2idx["<PAD>"]) for s in sentences_idx]
    tags_pad = [pad_sequence(t, max_len, vocab.tag2idx["<PAD>"]) for t in tags_idx]
    
    sentences_var = Variable(torch.LongTensor(sentences_pad))
    tags_var = Variable(torch.LongTensor(tags_pad))
    seq_lens_var = Variable(torch.LongTensor(seq_lens))
    mask_var = Variable((sentences_var.data != vocab.word2idx["<PAD>"]).float())
    
    if use_gpu:
        sentences_var = sentences_var.cuda()
        tags_var = tags_var.cuda()
        seq_lens_var = seq_lens_var.cuda()
        mask_var = mask_var.cuda()
        
    return sentences_var, tags_var, seq_lens_var, mask_var

def train(args):
    logger.info("Initializing Training Pipeline...")
    vocab = Vocabulary()
    train_data = load_data(args.train_file, vocab, is_train=True)
    vocab.compile()
    
    logger.info(f"Loaded {len(train_data)} sentences. Vocab Size: {len(vocab.word2idx)}")
    
    model = BiLSTM_CRF(
        vocab_size=len(vocab.word2idx),
        tag_to_ix=vocab.tag2idx,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_gpu=args.cuda
    )
    
    if args.cuda:
        model = model.cuda()
        
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for i in range(0, len(train_data), args.batch_size):
            batch = train_data[i:i + args.batch_size]
            sentences_var, tags_var, seq_lens_var, mask_var = prepare_batch(batch, vocab, args.cuda)
            
            optimizer.zero_grad()
            loss = model.neg_log_likelihood(sentences_var, seq_lens_var, tags_var, mask_var)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.data[0]
            
        logger.info(f"Epoch {epoch + 1}/{args.epochs} | Avg Loss: {epoch_loss / len(train_data):.4f}")
        
    if args.save_model:
        torch.save({'state_dict': model.state_dict(), 'vocab': vocab}, args.save_model)
        logger.info(f"Model saved to {args.save_model}")

def infer(args):
    if not args.text:
        raise ValueError("Inference mode requires --text argument.")
        
    logger.info("Booting Stanford CoreNLP and spaCy...")
    analyzer = DeepLinguisticAnalyzer(corenlp_path=args.corenlp_path)
    
    logger.info("Running Linguistic Analysis...")
    analysis_results = analyzer.analyze(args.text)
    
    logger.info("Linguistic Properties Extracted:")
    for cluster_id, mentions in analysis_results.get("coreference", {}).items():
        logger.info(f"  Coref Cluster {cluster_id}: {len(mentions)} mentions found.")
    
    if not args.load_model:
        logger.warning("No PyTorch checkpoint provided! Using initialized random weights for demo purpose.")
        vocab = Vocabulary()
        for token in analysis_results["spacy_syntax"]: vocab.add_word(token["text"])
        vocab.compile()
        
        model = BiLSTM_CRF(len(vocab.word2idx), vocab.tag2idx, args.embedding_dim, args.hidden_dim, use_gpu=args.cuda)
    else:
        checkpoint = torch.load(args.load_model)
        vocab = checkpoint['vocab']
        model = BiLSTM_CRF(len(vocab.word2idx), vocab.tag2idx, args.embedding_dim, args.hidden_dim, use_gpu=args.cuda)
        model.load_state_dict(checkpoint['state_dict'])
        
    if args.cuda: model = model.cuda()
    model.eval()
    
    tokens = [t["text"] for t in analysis_results["spacy_syntax"]]
    batch_data = [{"tokens": tokens, "tags": ["O"] * len(tokens)}]
    
    sentences_var, _, seq_lens_var, mask_var = prepare_batch(batch_data, vocab, args.cuda)
    
    scores, paths = model(sentences_var, seq_lens_var, mask_var)
    
    predicted_tags = [vocab.idx2tag[idx] for idx in paths[0]]
    logger.info("\nInference Output:")
    for token, tag in zip(tokens, predicted_tags):
        print(f"{token:15} \t {tag}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IE Pipeline 2017 - Training & Inference Orchestrator")
    parser.add_argument("--mode", type=str, choices=['train', 'infer'], required=True, help="Mode to run the script in.")
    parser.add_argument("--train_file", type=str, help="Path to JSONL training data.")
    parser.add_argument("--text", type=str, help="Raw text for inference mode.")
    parser.add_argument("--corenlp_path", type=str, default="/opt/stanford-corenlp-full-2018-02-27", help="Path to CoreNLP dir.")
    parser.add_argument("--save_model", type=str, default="model.pth", help="Checkpoint save path.")
    parser.add_argument("--load_model", type=str, help="Checkpoint load path.")
    
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA operations")
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    if args.cuda:
        logger.info("CUDA Enabled. Tensors will be allocated on GPU.")

    if args.mode == "train":
        train(args)
    elif args.mode == "infer":
        infer(args)