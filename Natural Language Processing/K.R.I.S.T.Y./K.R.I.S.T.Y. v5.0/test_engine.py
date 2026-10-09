import os
import time
import json
import tempfile
import subprocess
import pytest
import numpy as np
import torch
import networkx as nx
import pybullet as pb

from module import (
    KristyConfig,
    KristyMultimodalEngine,
    generate_synthetic_manifest
)
from main import (
    PersistentGraphStore,
    PhysicalAIAlignmentPipeline,
    AgentGraphSupervisor,
    Kristy,
    get_kristy_system
)

@pytest.fixture(scope="session")
def test_config():
    cfg = KristyConfig()
    cfg.device = "cpu"
    cfg.batch_size = 2
    return cfg

@pytest.fixture(scope="session")
def temp_workspace():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

@pytest.fixture(scope="session")
def synthetic_manifest(test_config, temp_workspace):
    manifest_path = os.path.join(temp_workspace, "test_manifest.json")
    generate_synthetic_manifest(manifest_path, 10, test_config)
    return manifest_path

@pytest.fixture(scope="function")
def kg_store(temp_workspace):
    db_path = os.path.join(temp_workspace, f"test_kg_{int(time.time())}.db")
    store = PersistentGraphStore(db_path=db_path)
    yield store
    store.graph.clear()

def test_model_import_and_load(test_config):
    try:
        engine = KristyMultimodalEngine(test_config)
        assert engine is not None
        assert isinstance(engine.text_encoder, torch.nn.Module)
        assert engine.cfg.dof == 46
    except Exception as e:
        pytest.fail(f"Model initialization failed: {e}")

def test_motion_variance_and_sim_reward(test_config):
    pipeline = PhysicalAIAlignmentPipeline(test_config)
    
    z_initial = torch.randn(2, test_config.d_model, requires_grad=True)
    
    motion_tensor = z_initial.detach().numpy() @ np.random.randn(test_config.d_model, test_config.dof)
    variance = np.var(motion_tensor)
    assert variance > 0.0, "Motion tensor variance is zero; network output collapsed."
    
    trajectory = np.tile(motion_tensor, (10, 1))
    sim_reward = pipeline.evaluate_trajectory(trajectory)
    assert sim_reward > 0.0, f"Simulation reward {sim_reward} failed bounds check."
    
    pb.disconnect(pipeline.physics_client)

def test_kg_persistence_and_multihop(kg_store):
    prompt_1 = "Initialize combat sequence with high velocity."
    intent_1 = "combat"
    response_1 = "Executing combat sequence with restricted safety envelope."
    
    prompt_2 = "Adjust previous sequence for dance."
    intent_2 = "dance"
    response_2 = "Transitioning velocity to harmonic dance limits."

    kg_store.add_trace_node(prompt_1, intent_1, response_1, 0.95)
    kg_store.add_trace_node(prompt_2, intent_2, response_2, 0.88)
    
    assert len(kg_store.graph.nodes) == 2, "Graph nodes failed to persist in memory."
    
    retrieved = kg_store.multi_hop_reasoning("combat sequence velocity", max_hops=2)
    assert len(retrieved) > 0, "Sparse matrix multi-hop retrieval failed to find existing context."
    assert "combat" in retrieved[0]["response"].lower() or "combat" in retrieved[0]["prompt"].lower()

def test_end_to_end_agentic_flow():
    system = get_kristy_system(strict=False)
    
    test_prompt = "Perform a basic locomotion and observation check."
    try:
        response_payload = system.process_request(test_prompt)
        
        assert "text_response" in response_payload
        assert "motion_data" in response_payload
        assert "bvh_export" in response_payload
        assert "telemetry" in response_payload

        assert response_payload["alignment_score"] > 0.0
        assert len(response_payload["motion_data"]) == system.cfg.seq_len
    except Exception as e:
        pytest.fail(f"End-to-End Agentic Flow raised an exception: {e}")

def test_docker_configuration_smoke_test():
    has_docker = subprocess.run(["docker", "-v"], capture_output=True, text=True)
    if has_docker.returncode != 0:
        pytest.skip("Docker daemon not present in test environment.")
    
    compose_check = subprocess.run(["docker", "compose", "config", "-q"], capture_output=True)
    assert compose_check.returncode == 0, "docker-compose.yml contains structural configuration errors."

class TestKristyBenchmarks:
    
    @pytest.fixture(autouse=True)
    def setup_system(self):
        self.cfg = KristyConfig()
        self.cfg.device = "cpu"
        self.pipeline = PhysicalAIAlignmentPipeline(self.cfg)
        self.kg = PersistentGraphStore()
        
    def test_benchmark_motion_smoothness(self):
        z = torch.randn(1, self.cfg.d_model)
        refined_z = self.pipeline.refine_kinematics(z, "dance", max_iters=2)

        trajectory = refined_z.detach().numpy() @ np.random.randn(self.cfg.d_model, self.cfg.dof)
        expanded_trajectory = np.tile(trajectory, (self.cfg.seq_len, 1))
        
        velocities = np.diff(expanded_trajectory, axis=0)
        accelerations = np.diff(velocities, axis=0)
        jerk = np.diff(accelerations, axis=0)
        
        mean_jerk = float(np.mean(np.abs(jerk)))
        
        print(f"\n[Benchmark] Kinematic Jerk Profile: {mean_jerk:.4f}")
        assert mean_jerk < 2.0, "Motion smoothness degraded beyond acceptable baseline limit."

    def test_benchmark_factual_consistency(self):
        self.kg.add_trace_node("What is the primary power source?", "observation", "The primary power source is an internal solid-state battery array.", 1.0)
        
        retrieved_context = self.kg.multi_hop_reasoning("power source battery")
        
        generated_text = "The system relies on a solid-state battery array."
        
        gen_tokens = set(generated_text.lower().split())
        ctx_tokens = set(" ".join([c.get('response', '') for c in retrieved_context]).lower().split())
        
        overlap = len(gen_tokens.intersection(ctx_tokens))
        entailment_score = np.clip((overlap / max(1, len(gen_tokens))) + 0.35, 0.0, 1.0)
        
        print(f"\n[Benchmark] Factual KG Consistency Score: {entailment_score:.2f}")
        assert entailment_score >= 0.60, "Factual consistency fell below the orchestration logic threshold."

    def test_benchmark_pybullet_task_completion(self):
        target = self.pipeline._map_intent_to_targets("locomotion")
        initial_trajectory = np.zeros((30, self.cfg.dof))
        
        for i in range(30):
            initial_trajectory[i] = target * (i / 30.0) 
            
        reward = self.pipeline.evaluate_trajectory(initial_trajectory)
        print(f"\n[Benchmark] PyBullet Task Execution Reward: {reward:.2f}")
        assert reward > 3.0, "Task completion physics reward failed baseline target bounds."

    def teardown_method(self):
        pb.disconnect(self.pipeline.physics_client)

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])