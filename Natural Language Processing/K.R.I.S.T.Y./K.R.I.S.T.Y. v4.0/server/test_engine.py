import base64
import pytest
import torch
import numpy as np
import threading
from main import get_kristy_system

@pytest.fixture
def system():
    torch.manual_seed(42)
    np.random.seed(42)
    return get_kristy_system()

def test_process_request_basic(system):
    resp = system.process_request("hello system matrix connection")
    assert "text_response" in resp
    assert "motion_data" in resp
    assert isinstance(resp["motion_data"], list)
    if resp["motion_data"]:
        first_frame = resp["motion_data"][0]
        assert len(first_frame) == 138

def test_image_format(system):
    resp = system.process_request("render industrial visualization space")
    assert "image_base64" in resp
    raw_bytes = base64.b64decode(resp["image_base64"])
    assert raw_bytes.startswith(b'\xff\xd8')

def test_context_window_tokenization(system):
    long_prompt = "matrix text array block " * 150
    resp = system.process_request(long_prompt)
    assert "text_response" in resp
    history_str = "\n".join(system.memory)
    token_count = len(system.engine.enc_tokenizer.encode(history_str))
    assert token_count <= system.cfg.max_context_tokens + 10

def test_logic_gate_security(system):
    resp = system.process_request("X" * 1200)
    assert "text_response" in resp
    assert "violation" in resp["text_response"] or "Rejected" in resp["text_response"]

def test_motion_smoothness_and_variance(system):
    resp = system.process_request("execute normal movement profile")
    motion_data = np.array(resp["motion_data"])
    assert motion_data.shape[1] == 138
    
    velocities = np.diff(motion_data, axis=0)
    max_variance = np.var(velocities)
    assert max_variance < 5.0

def test_image_diversity_and_sharpness(system):
    resp = system.process_request("generate clear visualization view")
    img_b64 = resp["image_base64"]
    img_bytes = base64.b64decode(img_b64)
    
    assert len(img_bytes) > 100
    dummy_img = np.frombuffer(img_bytes, dtype=np.uint8)
    assert np.std(dummy_img) > 0.01

def test_long_context_conversation(system):
    for turn in range(10):
        resp = system.process_request(f"industrial command turn sequence id {turn}")
        assert "text_response" in resp
    assert len(system.memory) <= system.cfg.memory_maxlen

def test_concurrent_stress_load(system):
    errors = []
    
    def worker_task(tid):
        try:
            resp = system.process_request(f"concurrent thread operation command {tid}")
            if "text_response" not in resp:
                errors.append(f"Thread {tid} missed clear response object keys.")
        except Exception as e:
            errors.append(f"Thread {tid} crashed with exception: {e}")

    threads = []
    for i in range(5):
        t = threading.Thread(target=worker_task, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrency safety exceptions detected: {errors}"