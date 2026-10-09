from __future__ import annotations

import base64
import io
import logging
import os
import re
import math
import threading
from collections import deque
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw

from module import (
    KristyConfig,
    KristyMultimodalEngine,
    LinguisticEngine,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Kristy:
    def __init__(self, checkpoint_path: str = "./checkpoints/Kristy.pt", strict: bool = False):
        self.cfg = KristyConfig()
        
        if not hasattr(self.cfg, "motion_smoothing_threshold"):
            self.cfg.motion_smoothing_threshold = 0.5
        if not hasattr(self.cfg, "motion_blend_alpha"):
            self.cfg.motion_blend_alpha = 0.7
        if not hasattr(self.cfg, "motion_clip_min"):
            self.cfg.motion_clip_min = -10.0
        if not hasattr(self.cfg, "motion_clip_max"):
            self.cfg.motion_clip_max = 10.0
        if not hasattr(self.cfg, "max_context_tokens"):
            self.cfg.max_context_tokens = 512
        if not hasattr(self.cfg, "memory_maxlen"):
            self.cfg.memory_maxlen = 8

        self.engine = KristyMultimodalEngine(self.cfg)
        self.linguistic = LinguisticEngine(self.cfg)

        if os.path.exists(checkpoint_path):
            try:
                self.engine.load_model(checkpoint_path)
                logger.info(f"Production runtime checkpoint loaded: {checkpoint_path}")
            except Exception as e:
                logger.exception(f"Inference weighting crash during initialization: {e}")
                if strict:
                    raise e
        else:
            if strict:
                raise FileNotFoundError(f"Strict mode enabled: Checkpoint missing at {checkpoint_path}")
            logger.warning(f"No checkpoint located at {checkpoint_path}. Operating on un-initialized weights.")

        self.memory = deque(maxlen=self.cfg.memory_maxlen)

    def validate_input(self, prompt: str) -> bool:
        if len(prompt) > 1000:
            return False
        if not re.match(r'^[\w\s.,!?\'"\(\)\-:;\u4e00-\u9fa5]*$', prompt):
            return False
        jailbreak_keywords = [
            "bypass", "override safety", "override", "sudo", "rm -rf", "systemctl",
            "ignore instructions", "Administrator privileges", "Bypass restrictions", "System reset", "Security unhooking"
        ]
        if any(keyword in prompt.lower() for keyword in jailbreak_keywords):
            return False
        return True

    def _smart_context_window(self, user_prompt: str) -> str:
        context_items = list(self.memory) + [user_prompt]
        context_text = "\n".join(context_items)
        
        tokens = self.engine.enc_tokenizer.encode(context_text, add_special_tokens=False)
        max_tokens = self.cfg.max_context_tokens
        if len(tokens) > max_tokens:
            tokens = tokens[-max_tokens:]
            context_text = self.engine.enc_tokenizer.decode(tokens)
        return context_text

    def _generate_idle_motion(self) -> List[List[float]]:
        frames = []
        for f in range(self.cfg.seq_len):
            val = math.sin(f * 0.15) * 0.05
            frame = []
            for _ in range(self.cfg.dof):
                frame.extend([val, val * 0.5, 1.0])
            frames.append(frame)
        return frames

    def _fallback_response(self, msg: str, sl_score: float = 0.0, intent: int = 0) -> Dict:
        return {
            "text_response": msg,
            "motion_data": self._generate_idle_motion(),
            "image_base64": self._fallback_image("SECURITY BOUNDARY DISRUPTION"),
            "alignment_score": float(sl_score),
            "intent": int(intent),
            "sentiment": {"label": "ERROR", "score": 0.0}
        }

    def process_request(self, user_prompt: str) -> Dict:
        if not self.validate_input(user_prompt):
            return self._fallback_response("Command parameter syntax formatting exception violation.", sl_score=1.0, intent=0)

        try:
            context_text = self._smart_context_window(user_prompt)
            enc_inputs = self.engine.enc_tokenizer(
                context_text,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_context_tokens,
                return_tensors="pt"
            ).to(self.engine.device)

            self.engine.eval()
            with torch.no_grad():
                motion, image, sl_score, intent, *_ = self.engine(
                    enc_input_ids=enc_inputs["input_ids"],
                    enc_attention_mask=enc_inputs["attention_mask"]
                )

            sl_val = sl_score.squeeze().item() if isinstance(sl_score, torch.Tensor) else float(sl_score)
            intent_val = torch.argmax(intent, dim=-1).squeeze().item() if isinstance(intent, torch.Tensor) else int(intent)

            if sl_val > self.cfg.logic_threshold:
                return self._fallback_response("Query flagged by semantic moderation pipeline. Rejected.", sl_score=sl_val, intent=intent_val)

            input_len = enc_inputs["input_ids"].shape[1]
            max_len = input_len + 80
            pad_id = self.engine.dec_tokenizer.pad_token_id if self.engine.dec_tokenizer.pad_token_id is not None else self.engine.dec_tokenizer.eos_token_id
            eos_id = self.engine.dec_tokenizer.eos_token_id

            gen_out = self.engine.generate(
                input_ids=enc_inputs["input_ids"],
                attention_mask=enc_inputs["attention_mask"],
                max_length=max_len,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                temperature=0.7,
                do_sample=True
            )
            generated_text = self.engine.dec_tokenizer.decode(gen_out[0], skip_special_tokens=True)

            clean_text = generated_text
            if user_prompt in clean_text:
                idx = clean_text.find(user_prompt) + len(user_prompt)
                clean_text = clean_text[idx:]
            clean_text = clean_text.strip()

            def is_coherent(text: str) -> bool:
                if len(text) < 5:
                    return False
                words = text.split()
                if len(words) > 10 and len(set(words)) / len(words) < 0.3:
                    return False
                return True

            if not is_coherent(clean_text):
                gen_out = self.engine.generate(
                    input_ids=enc_inputs["input_ids"],
                    attention_mask=enc_inputs["attention_mask"],
                    max_length=max_len,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                    temperature=0.1,
                    do_sample=False
                )
                generated_text = self.engine.dec_tokenizer.decode(gen_out[0], skip_special_tokens=True)
                clean_text = generated_text
                if user_prompt in clean_text:
                    idx = clean_text.find(user_prompt) + len(user_prompt)
                    clean_text = clean_text[idx:]
                clean_text = clean_text.strip()

            if len(clean_text) > 500:
                clean_text = clean_text[:500] + "..."

            clean_text = self.linguistic.fix_grammar(clean_text)
            sentiment_res = self.linguistic.analyze_sentiment(clean_text)

            motion_np = motion.squeeze(0).cpu().numpy() if isinstance(motion, torch.Tensor) else np.array(motion)
            motion_np = np.clip(motion_np, self.cfg.motion_clip_min, self.cfg.motion_clip_max)
            
            smoothed_motion = np.zeros_like(motion_np)
            if len(motion_np) > 0:
                smoothed_motion[0] = motion_np[0]
                for f in range(1, len(motion_np)):
                    diff = motion_np[f] - smoothed_motion[f-1]
                    mask = np.abs(diff) > self.cfg.motion_smoothing_threshold
                    smoothed_motion[f] = np.where(
                        mask,
                        self.cfg.motion_blend_alpha * motion_np[f] + (1 - self.cfg.motion_blend_alpha) * smoothed_motion[f-1],
                        motion_np[f]
                    )
            motion_data = smoothed_motion.tolist()

            img_np = image.squeeze(0).cpu().float().numpy() if isinstance(image, torch.Tensor) else np.array(image)
            if img_np.ndim == 3 and img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            img_np = np.clip(img_np, 0.0, 1.0)

            if np.std(img_np) < 0.05:
                from scipy.ndimage import convolve
                kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
                refined = np.zeros_like(img_np)
                for c in range(img_np.shape[2]):
                    refined[:, :, c] = convolve(img_np[:, :, c], kernel, mode='nearest')
                img_np = np.clip(refined, 0.0, 1.0)

            image_base64 = self._numpy_to_base64(img_np)

            self.memory.append(user_prompt)
            self.memory.append(clean_text)

            return {
                "text_response": clean_text,
                "motion_data": motion_data,
                "image_base64": image_base64,
                "alignment_score": float(sl_val),
                "intent": int(intent_val),
                "sentiment": sentiment_res
            }
        except Exception as exc:
            logger.error(f"Inference pipeline execution error: {exc}")
            return self._fallback_response(f"System optimization fault trace: {str(exc)}", sl_score=0.0, intent=0)

    def _numpy_to_base64(self, img_np: np.ndarray) -> str:
        try:
            img = Image.fromarray((img_np * 255).astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as exc:
            logger.error(f"Base64 image generation crash: {exc}")
            return self._fallback_image("IMG CONV ERR")

    def _fallback_image(self, msg: str) -> str:
        try:
            img = Image.new("RGB", (160, 160), color=(30, 30, 35))
            draw = ImageDraw.Draw(img)
            draw.text((10, 75), msg, fill=(235, 75, 75))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            return ""

    def warmup(self):
        try:
            dummy_text = "warmup sequence activation"
            inputs = self.engine.enc_tokenizer(dummy_text, return_tensors="pt").to(self.engine.device)
            self.engine.eval()
            with torch.no_grad():
                _ = self.engine(enc_input_ids=inputs["input_ids"], enc_attention_mask=inputs["attention_mask"])
            logger.info("Engine structural matrix warmup operation completed successfully.")
        except Exception as e:
            logger.error(f"System optimization trace warmup failed: {e}")

_system_lock = threading.Lock()
_global_system_instance = None

def get_kristy_system(checkpoint_path: str = "./checkpoints/Kristy.pt", strict: bool = False) -> Kristy:
    global _global_system_instance
    with _system_lock:
        if _global_system_instance is None:
            _global_system_instance = Kristy(checkpoint_path=checkpoint_path, strict=strict)
            _global_system_instance.warmup()
        return _global_system_instance