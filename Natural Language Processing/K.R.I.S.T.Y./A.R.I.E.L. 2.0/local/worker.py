import os
import json
from celery import Celery
from kg_engine import KnowledgeEngine

BROKER_URL = "pyamqp://guest@localhost//"

celery_app = Celery("ariel_tasks", broker=BROKER_URL)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

kb = KnowledgeEngine(db_path="knowledge_base.db")

@celery_app.task
def fetch_context_async(entity_name: str) -> str:
    """
    Return a concise, human‑readable string of facts
    associated with *entity_name*.
    """
    if not entity_name:
        return ""

    facts = kb.query_fact(entity_name)
    if not facts:
        return ""

    return " | ".join(f"{f['key']}: {f['value']}" for f in facts)
