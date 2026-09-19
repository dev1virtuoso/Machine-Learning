# main.py
"""
主入口：检索 → 图推理 → 生成 → 语法纠错
"""

import logging
import os                     # <‑‑ 新增
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from kristy_arch import KristyEngine, KristyLogitWarper, KristyConfig
from linguistic_engine import LinguisticEngine
from retrieval import BM25Retriever
from torch_geometric.data import Data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

class KristySystem2019:
    def __init__(self, model_path: str = "kristy_v3.0.pth", index_path: str = "index.jsonl"):
        self.config = KristyConfig()
        # tokenizer 与 generator 共享同一 vocab
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.tokenizer_name)

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 检索器
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"索引文件 {index_path} 不存在")
        self.retriever = BM25Retriever(index_path, device=self.device)

        # 主模型
        self.model = KristyEngine(self.config).to(self.device)
        try:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            logging.info(f"Loaded checkpoint: {model_path}")
        except Exception:
            logging.warning("Checkpoint not found, using pre‑trained BERT weights")

        self.model.eval()

        # 语言工具
        self.ling = LinguisticEngine()

        # 生成器（T5‑small） + LogitsProcessor
        self.generator = T5ForConditionalGeneration.from_pretrained(
            self.config.tokenizer_name
        ).to(self.device)
        self.logit_warper = KristyLogitWarper(
            self.model, threshold=self.config.logic_threshold
        )

    # ---------- 构造最简知识图 ----------
    def _build_graph_from_text(self, text: str) -> Data:
        """
        用词共现构造最简知识图（自环 + 共现边）。  
        由于 2019 版无高阶图特征，使用随机向量做节点特征，保证至少一条边。
        """
        tokens = text.split()
        size = len(tokens)
        idx_map = {w: i for i, w in enumerate(tokens)}

        # 共现窗口 2
        edges = []
        for i, wi in enumerate(tokens):
            for j, wj in enumerate(tokens):
                if i != j and (wi in wj or wj in wi):
                    edges.append([i, j])
        if not edges:
            edges = [[i, i] for i in range(size)]  # 至少自环

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # 随机节点特征
        x = torch.randn((size, self.config.hidden_size), device=self.device)
        return Data(x=x, edge_index=edge_index)

    # ---------- 查询处理 ----------
    def handle_query(self, text: str) -> str:
        """
        处理单条文本查询。
        1. 实体抽取
        2. 检索 top‑k 片段
        3. 构造知识图结构
        4. 前向推理得到逻辑门分数
        5. 若超阈值则抑制输出
        6. 生成回答（T5‑small + LogitsProcessor）
        7. 语法纠错
        """
        try:
            # 1. 实体抽取
            entities = self.ling.extract_entities(text)
            logging.debug(f"Entities: {entities}")

            # 2. 检索 top‑k 片段
            top_k = self.retriever.get_top_k(text, k=self.config.retrieval_k, return_scores=True)
            logging.debug(f"Top‑k retrieved: {top_k}")

            # 3. 构造知识图
            context = " ".join([t[0] for t in top_k])
            knowledge_graph = self._build_graph_from_text(context)

            # 4. 编码
            inputs = self.tokenizer.encode_plus(
                text,
                return_tensors="pt",
                max_length=self.config.max_seq_length,
                truncation=True,
                padding="max_length",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 5. 前向推理（BERT + Graph）
            with torch.no_grad():
                logits, logic_score = self.model(
                    **inputs, knowledge_graph=knowledge_graph
                )
            risk = logic_score.squeeze().cpu().item()
            logging.debug(f"Logic score: {risk:.4f}")

            # 6. 判断是否抑制
            if risk > self.config.logic_threshold:
                return (
                    "WARNING: Logic anomaly detected. "
                    "Output suppressed for factual safety."
                )

            # 7. 生成回答（T5‑small + LogitsProcessor）
            prompt = f"Answer the following question: {text}\nContext: {context}"
            prompt_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.config.max_seq_length,
                truncation=True,
                padding="max_length",
            )
            for k, v in prompt_ids.items():
                prompt_ids[k] = v.to(self.device)

            generated_ids = self.generator.generate(
                input_ids=prompt_ids["input_ids"],
                attention_mask=prompt_ids["attention_mask"],
                logits_processor=[self.logit_warper],
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )
            raw_answer = self.tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )

            # 8. 语法纠错
            return self.ling.polish_response(raw_answer)

        except Exception as exc:
            logging.exception(f"handle_query error: {exc}")
            return "ERROR: Unable to process the request."

if __name__ == "__main__":
    kristy = KristySystem2019()
    user_in = "Explain the fiscal results found in the latest JSON reports."
    print(f"USER: {user_in}")
    print(f"KRISTY: {kristy.handle_query(user_in)}")
