import argparse
import os
import random
import pickle
import mmap
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from module import Vocabulary, Encoder, Decoder, Seq2Seq, StyleDiscriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiskIndexedALLESDataset(Dataset):
    def __init__(self, src_path, trg_path, lang_path, style_path, vocab, max_len=50, is_pretrain=False):
        self.paths = {'src': src_path, 'trg': trg_path, 'lang': lang_path, 'style': style_path}
        self.vocab = vocab
        self.max_len = max_len
        self.is_pretrain = is_pretrain
        self.noise_ratio = 0.15
        
        self.offsets = {k: self._build_line_offsets(v) for k, v in self.paths.items()}
        self.mmaps = {k: None for k in self.paths.keys()}
        self.files = {k: None for k in self.paths.keys()}
        assert len(self.offsets['src']) == len(self.offsets['trg'])

    def _build_line_offsets(self, file_path):
        offsets = []
        with open(file_path, 'rb') as f:
            offsets.append(0)
            while f.readline():
                offsets.append(f.tell())
        return offsets[:-1]

    def _init_mmap(self, key):
        if self.mmaps[key] is None:
            self.files[key] = open(self.paths[key], 'r', encoding='utf-8')
            self.mmaps[key] = mmap.mmap(self.files[key].fileno(), 0, access=mmap.ACCESS_READ)

    def _get_line(self, key, idx):
        self._init_mmap(key)
        start_offset = self.offsets[key][idx]
        self.mmaps[key].seek(start_offset)
        line_bytes = self.mmaps[key].readline()
        return line_bytes.decode('utf-8').strip()

    def _tokenize(self, sentence):
        tokens = sentence.split()[:self.max_len-2]
        indices = [self.vocab.word2index.get(token, self.vocab.word2index["<unk>"]) for token in tokens]
        return [self.vocab.word2index["<sos>"]] + indices + [self.vocab.word2index["<eos>"]]

    def __len__(self):
        return len(self.offsets['src'])

    def __getitem__(self, idx):
        src_line = self._get_line('src', idx)
        trg_line = self._get_line('trg', idx)
        lang_id = int(self._get_line('lang', idx))
        style_id = int(self._get_line('style', idx))
        
        src_idx = self._tokenize(src_line)
        trg_idx = self._tokenize(trg_line)
        
        if self.is_pretrain:
            trg_idx = src_idx.copy() 
            src_idx = self._apply_denoising_noise(src_idx)
            
        return src_idx, trg_idx, lang_id, style_id, idx

    def _apply_denoising_noise(self, indices):
        noised = []
        core_indices = indices[1:-1]
        for idx in core_indices:
            prob = random.random()
            if prob < self.noise_ratio: 
                continue
            elif prob < (self.noise_ratio + 0.15): 
                noised.append(self.vocab.word2index["<unk>"])
            else: 
                noised.append(idx)
        if len(noised) == 0: 
            noised = core_indices.copy()
        return [indices[0]] + noised + [indices[-1]]

    def close(self):
        for k in self.mmaps.keys():
            if self.mmaps[k] is not None:
                self.mmaps[k].close()
            if self.files[k] is not None:
                self.files[k].close()

def collate_fn(batch):
    sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    src_batch_raw, trg_batch_raw, lang_batch_raw, style_batch_raw, global_indices = zip(*sorted_batch)
    
    src_lengths = torch.tensor([len(x) for x in src_batch_raw])
    src_batch = nn.utils.rnn.pad_sequence([torch.tensor(x) for x in src_batch_raw], padding_value=0, batch_first=True)
    trg_batch = nn.utils.rnn.pad_sequence([torch.tensor(x) for x in trg_batch_raw], padding_value=0, batch_first=True)
    
    return src_batch, src_lengths, trg_batch, torch.tensor(lang_batch_raw), torch.tensor(style_batch_raw), torch.tensor(global_indices)

def scan_metadata_limits(lang_path, style_path):
    max_lang, max_style = 0, 0
    with open(lang_path, 'r') as fl, open(style_path, 'r') as fs:
        for l, s in zip(fl, fs):
            max_lang = max(max_lang, int(l.strip()))
            max_style = max(max_style, int(s.strip()))
    return max_lang + 1, max_style + 1

class PersistentMemoryBank:
    def __init__(self, size, hidden_size, max_history=5):
        self.max_history = max_history
        self.bank = torch.zeros(size, max_history, hidden_size)
        self.history_lens = torch.zeros(size, dtype=torch.long)

    def get_context(self, indices, target_device):
        return self.bank[indices].to(target_device), self.history_lens[indices].to(target_device)

    def update_context(self, indices, new_context):
        for i, idx in enumerate(indices.tolist()):
            current_len = self.history_lens[idx].item()
            if current_len < self.max_history:
                self.bank[idx, current_len] = new_context[i].detach().cpu()
                self.history_lens[idx] += 1
            else:
                self.bank[idx, :-1] = self.bank[idx, 1:].clone()
                self.bank[idx, -1] = new_context[i].detach().cpu()

class CurriculumScheduler:
    def __init__(self, initial_max_len=15, progression_step=5):
        self.current_max_len = initial_max_len
        self.progression_step = progression_step

    def update_stage(self, epoch, dataset):
        self.current_max_len += (epoch * self.progression_step)
        dataset.max_len = min(120, self.current_max_len)
        dataset.noise_ratio = max(0.05, 0.25 - (epoch * 0.02))

class EWC:
    def __init__(self, model, dataloader, criterion):
        self.model = model
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.saved_params = {n: p.clone().detach() for n, p in self.params.items()}
        self.fisher = self._compute_fisher(dataloader, criterion)

    def _compute_fisher(self, dataloader, criterion):
        fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        self.model.eval()
        
        for i, batch_data in enumerate(dataloader):
            if i > 20: 
                break 
            src, lengths, trg, lang, style, global_indices = batch_data
            src, lengths, trg, lang, style = src.to(device), lengths.to(device), trg.to(device), lang.to(device), style.to(device)
            
            self.model.zero_grad()
            batch_size = src.size(0)
            isolated_memory = torch.zeros(batch_size, 5, self.model.decoder.rnn.hidden_size).to(device)
            isolated_lens = torch.zeros(batch_size, dtype=torch.long).to(device)
            
            output, _, _ = self.model(src, lengths, trg, lang, style, memory_tensor=isolated_memory, history_lens=isolated_lens)
            loss = criterion(output[:, 1:].reshape(-1, output.shape[-1]), trg[:, 1:].reshape(-1))
            loss.backward()
            
            for n, p in self.params.items():
                if p.grad is not None:
                    fisher[n].data += p.grad.data ** 2 / len(dataloader)
                    
        self.model.zero_grad()
        self.model.train()
        return fisher

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += torch.sum(self.fisher[n] * (p - self.saved_params[n]) ** 2)
        return loss

def calculate_bleu(predicted_batch, target_batch, pad_idx=0):
    total_bleu = 0.0
    count = 0
    for pred, trg in zip(predicted_batch, target_batch):
        p_tokens = [t.item() for t in pred if t.item() != pad_idx]
        t_tokens = [t.item() for t in trg if t.item() != pad_idx]
        if len(p_tokens) == 0 or len(t_tokens) == 0:
            continue
        
        match_count = 0
        for token in p_tokens:
            if token in t_tokens:
                match_count += 1
        total_bleu += (match_count / len(p_tokens))
        count += 1
    return total_bleu / count if count > 0 else 0.0

def train_epoch(model, discriminator, dataloader, memory_bank, optimizer, disc_optimizer, criterion, clip, ewc=None, is_pretrain=False, current_epoch=0):
    model.train()
    discriminator.train()
    epoch_loss = 0
    
    alpha = min(1.0, 0.1 + float(current_epoch) * 0.1)

    for i, (src, src_lengths, trg, lang, style, global_indices) in enumerate(dataloader):
        src, src_lengths, trg, lang, style = src.to(device), src_lengths.to(device), trg.to(device), lang.to(device), style.to(device)
        
        memory_tensor, history_lens = memory_bank.get_context(global_indices, device)

        if is_pretrain and i % 5 == 0:
            with torch.no_grad():
                pseudo_trg, _ = model.greedy_decode(src, src_lengths, lang, style, memory_tensor, history_lens, max_len=trg.size(1))
            trg = pseudo_trg 
        
        optimizer.zero_grad()
        disc_optimizer.zero_grad()
        
        output, raw_enc_hidden, new_mem_state = model(src, src_lengths, trg, lang, style, memory_tensor, history_lens)
        
        output_dim = output.shape[-1]
        loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
        
        if ewc is not None:
            loss += 5000.0 * ewc.penalty(model)
            
        style_preds = discriminator(raw_enc_hidden, alpha=alpha)
        adv_loss = nn.CrossEntropyLoss()(style_preds, style)
        
        style_cycle_preds = discriminator(new_mem_state, alpha=alpha)
        cycle_loss = nn.CrossEntropyLoss()(style_cycle_preds, style)
        
        total_gen_loss = loss + 0.1 * adv_loss + 0.05 * cycle_loss
        total_gen_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        if i % 2 == 0:
            disc_optimizer.zero_grad()
            style_preds_detached = discriminator(raw_enc_hidden.detach(), alpha=0.0)
            disc_loss = nn.CrossEntropyLoss()(style_preds_detached, style)
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip)
            disc_optimizer.step()
        
        memory_bank.update_context(global_indices, new_mem_state)
        epoch_loss += total_gen_loss.item()
        
    return epoch_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_data', type=str, required=True)
    parser.add_argument('--trg_data', type=str, required=True)
    parser.add_argument('--lang_data', type=str, required=True)
    parser.add_argument('--style_data', type=str, required=True)
    parser.add_argument('--vocab_path', type=str, default="shared_vocab.pkl")
    parser.add_argument('--task_mode', type=str, choices=['pretrain', 'fine_tune'], default='fine_tune')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    args = parser.parse_args()

    num_langs, num_styles = scan_metadata_limits(args.lang_data, args.style_data)
    print(f"Dynamic Metadata constraints identified: {num_langs} languages, {num_styles} styles.")

    if os.path.exists(args.vocab_path):
        vocab = pickle.load(open(args.vocab_path, 'rb'))
    else:
        vocab = Vocabulary("shared_global")
        vocab.build_vocab_from_file(args.src_data)
        pickle.dump(vocab, open(args.vocab_path, 'wb'))

    dataset = DiskIndexedALLESDataset(args.src_data, args.trg_data, args.lang_data, args.style_data, vocab, is_pretrain=(args.task_mode=='pretrain'))

    encoder = Encoder(len(vocab), 256, 512, 512, 0.4)
    decoder = Decoder(len(vocab), 256, 512, 512, num_langs, num_styles, 0.4)
    model = Seq2Seq(encoder, decoder, src_pad_idx=vocab.pad_idx, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx).to(device)
    
    discriminator = StyleDiscriminator(512, num_styles).to(device)
    memory_bank = PersistentMemoryBank(len(dataset), decoder.rnn.hidden_size)
    curriculum = CurriculumScheduler()

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    start_epoch = 0
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        discriminator.load_state_dict(checkpoint['disc_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state'])
        memory_bank.bank = checkpoint['memory_bank_state']
        memory_bank.history_lens = checkpoint['memory_bank_lens']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming execution protocol from epoch {start_epoch}")

    print("Initiating execution protocol...")
    for epoch in range(start_epoch, args.epochs):
        curriculum.update_stage(epoch, dataset)
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=2, drop_last=False)
        ewc_module = EWC(model, dataloader, criterion) if args.task_mode == 'fine_tune' else None
        
        loss = train_epoch(
            model=model, 
            discriminator=discriminator, 
            dataloader=dataloader, 
            memory_bank=memory_bank,
            optimizer=optimizer, 
            disc_optimizer=disc_optimizer, 
            criterion=criterion, 
            clip=1.0, 
            ewc=ewc_module, 
            is_pretrain=(args.task_mode=='pretrain'),
            current_epoch=epoch
        )
        print(f"[{args.task_mode.upper()}] Phase | Epoch: {epoch+1:02d} | Loss: {loss:.4f}")

        model.eval()
        with torch.no_grad():
            for src, src_lengths, trg, lang, style, global_indices in dataloader:
                src, src_lengths, trg, lang, style = src.to(device), src_lengths.to(device), trg.to(device), lang.to(device), style.to(device)
                m_tensor, h_lens = memory_bank.get_context(global_indices, device)
                preds, _ = model.greedy_decode(src, src_lengths, lang, style, m_tensor, h_lens, max_len=trg.size(1))
                bleu_score = calculate_bleu(preds, trg, pad_idx=vocab.pad_idx)
                print(f"Validation Sample Check | BLEU Metric Precision: {bleu_score:.4f}")
                break

        checkpoint_path = f"alles_{args.task_mode}_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'disc_state': discriminator.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'disc_optimizer_state': disc_optimizer.state_dict(),
            'memory_bank_state': memory_bank.bank,
            'memory_bank_lens': memory_bank.history_lens,
        }, checkpoint_path)

    dataset.close()

if __name__ == "__main__":
    main()