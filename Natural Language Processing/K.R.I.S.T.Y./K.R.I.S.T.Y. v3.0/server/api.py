from contextlib import asynccontextmanager
import logging
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torch_geometric.data import Data
from transformers import T5ForConditionalGeneration, T5Tokenizer

from kristy_arch import KristyConfig, KristyEngine, KristyLogitWarper
from linguistic_engine import LinguisticEngine
from retrieval import BM25Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

class AppState:
    config: KristyConfig
    tokenizer: T5Tokenizer
    device: torch.device
    retriever: BM25Retriever
    model: KristyEngine
    generator: T5ForConditionalGeneration
    logit_warper: KristyLogitWarper
    ling: LinguisticEngine

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting K.R.I.S.T.Y. server…")

    state.config = KristyConfig()
    state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state.tokenizer = T5Tokenizer.from_pretrained(state.config.tokenizer_name)

    index_path = os.getenv("BM25_INDEX", "index.jsonl")
    if not os.path.exists(index_path):
        raise RuntimeError(f"BM25 index not found: {index_path}")
    state.retriever = BM25Retriever(index_path, device=state.device)

    state.model = KristyEngine(state.config).to(state.device)
    checkpoint = "kristy_v3.0.pth"
    if os.path.exists(checkpoint):
        state.model.load_state_dict(
            torch.load(checkpoint, map_location=state.device)
        )
        logging.info(f"Checkpoint loaded: {checkpoint}")
    state.model.eval()

    state.generator = T5ForConditionalGeneration.from_pretrained(
        state.config.tokenizer_name
    ).to(state.device)
    state.logit_warper = KristyLogitWarper(
        state.model, threshold=state.config.logic_threshold
    )

    state.ling = LinguisticEngine()

    logging.info("K.R.I.S.T.Y. server ready.")
    yield


app = FastAPI(
    title="K.R.I.S.T.Y. API",
    description="K.R.I.S.T.Y. server",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str
    context: str | None = None


class QueryResponse(BaseModel):
    answer: str
    logic_score: float
    risk_flag: bool


def _build_graph(text: str) -> Data:
    tokens = text.split()
    size = len(tokens)
    if size == 0:
        return Data()
    idx = torch.arange(size, dtype=torch.long, device=state.device)
    edge_index = torch.cartesian_prod(idx, idx).t().contiguous()
    x = torch.randn((size, state.config.hidden_size), device=state.device)
    return Data(x=x, edge_index=edge_index)


@app.post("/infer", response_model=QueryResponse)
def infer(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if req.context:
        context = req.context
    else:
        top_k = state.retriever.get_top_k(
            req.question, k=state.config.retrieval_k, return_scores=True
        )
        context = " ".join([t[0] for t in top_k])

    graph = _build_graph(context)

    inputs = state.tokenizer(
        req.question,
        return_tensors="pt",
        max_length=state.config.max_seq_length,
        truncation=True,
        padding="max_length",
    )
    inputs = {k: v.to(state.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits, logic_score = state.model(**inputs, knowledge_graph=graph)
    risk = logic_score.squeeze().item()
    risk_flag = risk > state.config.logic_threshold

    if risk_flag:
        answer = (
            "Logic anomaly detected – output suppressed for safety.\n"
            f"(risk={risk:.3f})"
        )
    else:
        prompt = f"Answer the following question: {req.question}\nContext: {context}"
        prompt_ids = state.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=state.config.max_seq_length,
            truncation=True,
            padding="max_length",
        )
        prompt_ids = {k: v.to(state.device) for k, v in prompt_ids.items()}
        gen_ids = state.generator.generate(
            input_ids=prompt_ids["input_ids"],
            attention_mask=prompt_ids["attention_mask"],
            logits_processor=[state.logit_warper],
            max_length=128,
            num_beams=4,
            early_stopping=True,
        )
        raw_answer = state.tokenizer.decode(
            gen_ids[0], skip_special_tokens=True
        )
        answer = state.ling.polish_response(raw_answer)

    return QueryResponse(
        answer=answer,
        logic_score=risk,
        risk_flag=risk_flag,
    )