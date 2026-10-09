import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import BertModel
from transformers.generation.logits_process import LogitsProcessor
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

@dataclass
class KristyConfig:
    model_name: str = "bert-base-uncased"
    tokenizer_name: str = "t5-small"
    hidden_size: int = 768
    dropout_prob: float = 0.1
    learning_rate: float = 5e-5
    batch_size: int = 16
    train_epochs: int = 3
    pad_token_id: int = -100
    max_seq_length: int = 128
    logic_threshold: float = 0.7
    graph_layers: int = 2
    graph_hidden: int = 128
    graph_output_dim: int = 128
    retrieval_k: int = 5
    logits_processor: list = None

class KristyEngine(nn.Module):

    def __init__(self, config: KristyConfig):
        super().__init__()
        self.config = config

        self.bert = BertModel.from_pretrained(config.model_name)
        self.bert.config.output_hidden_states = False
        self.bert.config.return_dict = False

        self.reasoning_gate = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.graph_layers = nn.ModuleList()
        in_dim = config.hidden_size
        for _ in range(config.graph_layers):
            self.graph_layers.append(GCNConv(in_dim, config.graph_hidden))
            in_dim = config.graph_hidden
        self.graph_proj = nn.Linear(config.graph_hidden, config.hidden_size)

        self.output_head = nn.Linear(config.hidden_size, self.bert.config.vocab_size)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        knowledge_graph: Data = None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=False,
            return_dict=False,
        )
        last_hidden = outputs[0]
        pooled_output = outputs[1]

        logic_score = self.reasoning_gate(pooled_output)

        if knowledge_graph is not None:
            if (knowledge_graph.x is None) or (knowledge_graph.x.numel() == 0):
                x = pooled_output
                num_nodes = pooled_output.size(0)
                edge_index = torch.arange(
                    num_nodes, dtype=torch.long
                ).unsqueeze(0).repeat(2, 1).to(self.bert.device)
                knowledge_graph = Data(x=x, edge_index=edge_index)
            else:
                x = knowledge_graph.x

            for conv in self.graph_layers:
                x = conv(x, knowledge_graph.edge_index)
                x = nn.functional.relu(x)
            x = self.graph_proj(x)

            if x.dim() == 2:
                x = x.unsqueeze(1)
            x = x.expand_as(last_hidden)
            last_hidden = last_hidden + x

        logits = self.output_head(last_hidden)
        return logits, logic_score

class KristyLogitWarper(LogitsProcessor):

    def __init__(self, model: KristyEngine, threshold: float = 0.7):
        super().__init__()
        self.model = model
        self.threshold = threshold

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        with torch.no_grad():
            outputs = self.model.bert(
                input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            cls_hidden = outputs.pooler_output
            logic = self.model.reasoning_gate(cls_hidden).squeeze()
            factor = torch.clamp(1.0 - logic, 0.0, 1.0).unsqueeze(-1)
            factor = torch.max(factor, torch.tensor(0.1).to(factor.device))
            scores = scores * factor
        return scores
