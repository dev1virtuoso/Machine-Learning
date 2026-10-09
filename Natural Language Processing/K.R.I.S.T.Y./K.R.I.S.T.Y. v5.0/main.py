from __future__ import annotations

import base64
import io
import math
import os
import sqlite3
import threading
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import networkx as nx
from PIL import Image
import pybullet as pb
import pybullet_data

from module import KristyConfig, KristyMultimodalEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PyBulletManager:
    _client_id: Optional[int] = None
    _lock = threading.Lock()

    @classmethod
    def get_client(cls) -> int:
        with cls._lock:
            if cls._client_id is None or not pb.isConnected(physicsClientId=cls._client_id):
                cls._client_id = pb.connect(pb.DIRECT)
                pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cls._client_id)
            return cls._client_id

class PersistentGraphStore:
    def __init__(self, db_path: str = "./checkpoints/kristy_v5_knowledge_graph.db"):
        self.db_path = db_path
        self.graph = nx.DiGraph()
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()
        self._load_from_storage()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    node_id TEXT PRIMARY KEY, node_type TEXT, content TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_edges (
                    source TEXT, target TEXT, weight REAL
                )
            """)
            conn.commit()

    def _load_from_storage(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT node_id, node_type, content FROM graph_nodes")
            for nid, ntype, content in c.fetchall():
                self.graph.add_node(nid, type=ntype, content=content)
            c.execute("SELECT source, target, weight FROM graph_edges")
            for src, tgt, w in c.fetchall():
                self.graph.add_edge(src, tgt, weight=w)

    def insert_node(self, node_id: str, node_type: str, content: str):
        with self.lock:
            self.graph.add_node(node_id, type=node_type, content=content)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT OR REPLACE INTO graph_nodes VALUES (?, ?, ?)", (node_id, node_type, content))
                conn.commit()

    def insert_edge(self, source: str, target: str, weight: float = 1.0):
        with self.lock:
            self.graph.add_edge(source, target, weight=weight)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT OR REPLACE INTO graph_edges VALUES (?, ?, ?)", (source, target, weight))
                conn.commit()

    def multi_hop_reasoning(self, query: str, max_hops: int = 2) -> List[Dict[str, Any]]:
        with self.lock:
            if not self.graph.nodes:
                return []
            
            q_words = set(query.lower().split())
            best_node = None
            max_score = 0.0
            
            for nid, data in self.graph.nodes(data=True):
                c_words = set(data.get("content", "").lower().split())
                inter = q_words.intersection(c_words)
                score = len(inter) / max(1, math.sqrt(len(q_words) * len(c_words)))
                if score > max_score:
                    max_score = score
                    best_node = nid
                    
            if not best_node or max_score < 0.05:
                return []
                
            context = []
            visited = set()
            queue = [(best_node, 0)]
            while queue:
                curr, hop = queue.pop(0)
                if curr in visited or hop > max_hops:
                    continue
                visited.add(curr)
                
                ndata = self.graph.nodes[curr]
                context.append({"node_id": curr, "type": ndata.get("type"), "content": ndata.get("content")})
                for nxt in self.graph.neighbors(curr):
                    if nxt not in visited:
                        queue.append((nxt, hop + 1))
            return context

class PhysicalAIAlignmentPipeline:
    def __init__(self, cfg: KristyConfig):
        self.cfg = cfg
        self.client_id = PyBulletManager.get_client()

    def evaluate_trajectory(self, motion_sequence: np.ndarray) -> Tuple[float, np.ndarray]:
        cid = self.client_id
        pb.resetSimulation(physicsClientId=cid)
        pb.setGravity(0, 0, -9.81, physicsClientId=cid)
        
        plane_id = pb.loadURDF("plane.urdf", physicsClientId=cid)
        robot_id = pb.loadURDF("r2d2.urdf", [0, 0, 0.4], physicsClientId=cid)
        num_joints = pb.getNumJoints(robot_id, physicsClientId=cid)
        
        frames, dof = motion_sequence.shape
        refined_sequence = motion_sequence.copy()
        total_error = 0.0
        stability_penalty = 0.0
        
        for t in range(frames):
            for j in range(min(num_joints, dof)):
                pb.setJointMotorControl2(
                    bodyIndex=robot_id, jointIndex=j, controlMode=pb.POSITION_CONTROL,
                    targetPosition=float(motion_sequence[t, j]), physicsClientId=cid
                )
            pb.stepSimulation(physicsClientId=cid)
            
            pos, _ = pb.getBasePositionAndOrientation(robot_id, physicsClientId=cid)
            if pos[2] < 0.2:
                stability_penalty += 1.5
                
            for j in range(min(num_joints, dof)):
                js = pb.getJointState(robot_id, j, physicsClientId=cid)
                actual_pos = js[0]
                total_error += abs(actual_pos - motion_sequence[t, j])
                refined_sequence[t, j] = 0.8 * motion_sequence[t, j] + 0.2 * actual_pos
                
        alignment_score = max(0.1, 10.0 - (total_error * 0.02) - stability_penalty)
        return float(alignment_score), refined_sequence

class BvhMotionExporter:
    @staticmethod
    def export_to_string(motion_data: np.ndarray, seq_len: int, dof: int) -> str:
        bvh = [
            "HIERARCHY", "ROOT Hips", "{",
            "  OFFSET 0.00 0.00 0.00",
            "  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"
        ]
        num_joints = max(1, (dof - 6) // 3)
        indent = "  "
        for i in range(num_joints):
            bvh.append(f"{indent}JOINT Joint_{i}")
            bvh.append(f"{indent}{{")
            indent += "  "
            bvh.append(f"{indent}OFFSET 0.00 0.40 0.00")
            bvh.append(f"{indent}CHANNELS 3 Zrotation Xrotation Yrotation")
            
        bvh.append(f"{indent}End Site")
        bvh.append(f"{indent}{{")
        bvh.append(f"{indent}  OFFSET 0.00 0.10 0.00")
        bvh.append(f"{indent}}}")
        
        for _ in range(num_joints):
            indent = indent[:-2]
            bvh.append(f"{indent}}}")
        bvh.append("}")
        
        bvh.append("MOTION")
        bvh.append(f"Frames: {seq_len}")
        bvh.append("Frame Time: 0.033333")
        
        for t in range(seq_len):
            frame = [0.0, 0.0, 0.4]
            for d in range(min(dof, motion_data.shape[1])):
                frame.append(float(motion_data[t, d]) * 57.2958)
            while len(frame) < 6 + (num_joints * 3):
                frame.append(0.0)
            bvh.append(" ".join(f"{v:.4f}" for v in frame))
            
        return "\n".join(bvh)

class AgentGraphSupervisor:
    def build_kristy_agent_flow(self, plan: Dict[str, Any]) -> str:
        stages = ["INIT_COORDINATE", "GRAPHRAG_EXPANSION", "LATENT_MOE_ROUTING", "PYBULLET_CLOSE_LOOP", "PRODUCTION_AUDIT"]
        plan["execution_flow"] = " -> ".join(stages)
        return plan["execution_flow"]

    def execute_eval(self, motion_data: np.ndarray) -> Tuple[float, float]:
        jerk = float(np.mean(np.square(np.diff(motion_data, axis=0))))
        logic_score = max(0.0, min(1.0, 1.0 - jerk))
        return logic_score, 0.95

class Kristy:
    def __init__(self, cfg: KristyConfig):
        self.cfg = cfg
        self.kg_store = PersistentGraphStore()
        self.pipeline = PhysicalAIAlignmentPipeline(cfg)
        self.supervisor = AgentGraphSupervisor()
        self.engine = KristyMultimodalEngine(cfg)
        
        ckpt = "./checkpoints/kristy_best_weights.pt"
        if os.path.exists(ckpt):
            try:
                self.engine.load_state_dict(torch.load(ckpt, map_location=cfg.device))
                logger.info("Production neural states successfully re-synchronized.")
            except Exception as e:
                logger.error(f"Checkpoint parsing failure: {e}")

    def process_request(self, user_prompt: str) -> Dict[str, Any]:
        context = self.kg_store.multi_hop_reasoning(user_prompt)
        ctx_str = " ".join([c["content"] for c in context])
        
        augmented = f"Context: {ctx_str} Query: {user_prompt}" if ctx_str else user_prompt
        text_resp = self.engine.generate_text(augmented)
        
        tokens = self.engine.enc_tokenizer(user_prompt)
        ids = tokens["input_ids"].to(self.cfg.device)
        mask = tokens["attention_mask"].to(self.cfg.device)
        
        self.engine.eval()
        with torch.no_grad():
            outputs = self.engine(ids, mask)
            motion_np = outputs["motion_pred"].squeeze(0).cpu().numpy()
            img_tensor = outputs["image_pred"].squeeze(0).cpu()
            intent_idx = outputs["intent_pred"].argmax(dim=-1).item()
            
        intents = ["dance", "combat", "locomotion", "conversation", "observation"]
        prim_intent = intents[intent_idx] if intent_idx < len(intents) else "unknown"
        
        sim_score, refined_motion = self.pipeline.evaluate_trajectory(motion_np)
        plan = {"primary_intent": prim_intent, "target_frames": self.cfg.seq_len}
        self.supervisor.build_kristy_agent_flow(plan)
        l_score, k_score = self.supervisor.execute_eval(refined_motion)
        
        bvh_out = BvhMotionExporter.export_to_string(refined_motion, self.cfg.seq_len, self.cfg.dof)
        
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img_np).save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        self.kg_store.insert_node(f"node_{int(time.time())}", "trace", f"Prompt: {user_prompt} Response: {text_resp}")
        
        return {
            "text_response": text_resp,
            "motion_data": refined_motion.tolist(),
            "bvh_export": bvh_out,
            "image_base64": img_b64,
            "alignment_score": float(sim_score),
            "intent": prim_intent,
            "telemetry": {
                "cross_modal_consistency": float(l_score),
                "sim_reward": float(sim_score),
                "hop_count": len(context)
            }
        }

_global_instance = None
_lock = threading.Lock()

def get_kristy_system() -> Kristy:
    global _global_instance
    with _lock:
        if _global_instance is None:
            _global_instance = Kristy(KristyConfig())
        return _global_instance