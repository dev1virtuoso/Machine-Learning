import os
from typing import List, Optional

import torch
import torch.nn as nn
import spacy
from spacy.lang.en import English

class Seq2SeqGrammarRefiner(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        hidden_size: int = 512,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(
            embed_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.decoder = nn.LSTM(
            embed_size + hidden_size * 2, hidden_size, batch_first=True
        )
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        src: torch.LongTensor,
        tgt: Optional[torch.LongTensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        batch, seq_len = src.size()
        emb_src = self.embedding(src)
        enc_out, (h_n, c_n) = self.encoder(emb_src)

        h_0 = torch.tanh(h_n[0] + h_n[1]).unsqueeze(0)
        c_0 = torch.tanh(c_n[0] + c_n[1]).unsqueeze(0)

        max_len = tgt.size(1) if tgt is not None else seq_len
        outputs = torch.zeros(batch, max_len, self.out.out_features, device=src.device)

        dec_input = torch.full((batch, 1), 1, dtype=torch.long, device=src.device)

        for t in range(max_len):
            emb_dec = self.embedding(dec_input)
            enc_out_t = enc_out[:, t : t + 1, :]
            rnn_input = torch.cat((emb_dec, enc_out_t), dim=2)
            dec_out, (h_0, c_0) = self.decoder(rnn_input, (h_0, c_0))
            logits = self.out(dec_out.squeeze(1))
            outputs[:, t, :] = logits

            if tgt is not None:
                mask = torch.rand(batch, 1, device=src.device) < teacher_forcing_ratio
                dec_input = torch.where(mask, tgt[:, t : t + 1], logits.argmax(1).unsqueeze(1))

        return outputs

class LinguisticEngine:

    def __init__(
        self,
        model_path: str = "gec_seq2seq_2017.pth",
        vocab: Optional[dict[str, int]] = None,
    ) -> None:
        try:
            self.nlp = spacy.load("en_core_web_md")
        except Exception as exc:
            self.nlp = None
            print(f"spaCy model failed to load: {exc}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = vocab
        self.id_to_word = {v: k for k, v in vocab.items()} if vocab else {}

        vocab_size = len(vocab) if vocab else 30000
        self.grammar_model = Seq2SeqGrammarRefiner(vocab_size).to(self.device)

        if os.path.exists(model_path):
            self.grammar_model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
        self.grammar_model.eval()

    def fix_grammar(self, text: str, max_len: int = 32) -> str:
        if not self.nlp or not text:
            return text

        tokens = [tok.text for tok in self.nlp(text)]
        if self.vocab is None:
            return text

        token_ids = torch.tensor(
            [[self.vocab.get(tok, self.vocab.get("<unk>", 0)) for tok in tokens]],
            device=self.device,
        )

        if token_ids.size(1) < max_len:
            pad_len = max_len - token_ids.size(1)
            token_ids = torch.cat(
                [
                    token_ids,
                    torch.zeros(1, pad_len, dtype=torch.long, device=self.device),
                ],
                dim=1,
            )

        with torch.no_grad():
            out_logits = self.grammar_model(token_ids)

        pred_ids = out_logits.argmax(2).squeeze(0).cpu().tolist()
        corrected = [
            self.id_to_word.get(i, "")
            for i in pred_ids
            if i != self.vocab.get("<pad>", 0)
        ]
        return " ".join(corrected)

    def extract_entities(self, text: str) -> List[str]:
        if not self.nlp:
            return []
        doc = self.nlp(text)
        return [
            ent.text
            for ent in doc.ents
            if ent.label_ in {"PERSON", "ORG", "GPE", "PRODUCT"}
        ]
