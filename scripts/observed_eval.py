"""调用官方 evaluator；只读观察真实摄像头、推理设备和传入模拟器的动作。"""
from dataclasses import asdict
import hashlib
import json
import subprocess
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch
from lerobot.envs.libero import LiberoEnv
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts.lerobot_eval import main

out = Path(os.environ['BASELINE_EVIDENCE_DIR'])
out.mkdir(parents=True, exist_ok=True)
started = time.monotonic()
evidence = {'status': 'infrastructure_failure', 'success': None, 'env_steps': 0,
            'inference_calls': 0, 'condition': 'Baseline', 'machine_id': socket.gethostname(),
            'checkpoint_revision': '6721902bc4d61e50a3bfdb11dfb4cb626f05d102',
            'mask_geometry': None, 'target_coverage': None, 'seed': 42}
original_images = SmolVLAPolicy.prepare_images
original_reset = LiberoEnv.reset
original_step = LiberoEnv.step


def observed_images(self, batch):
    images, masks = original_images(self, batch)
    if evidence['inference_calls'] == 0:
        keys = [key for key in self.config.image_features if key in batch]
        evidence['input_keys'] = list(batch)
        evidence['image_keys'] = keys
        evidence['input_shapes'] = {k: list(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        evidence['parameter_devices'] = sorted({str(p.device) for p in self.parameters()})
        evidence['parameter_dtypes'] = sorted({str(p.dtype) for p in self.parameters()})
        evidence['inference_settings'] = asdict(self.config)
        evidence['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        evidence['config_hash'] = hashlib.sha256(json.dumps(asdict(self.config), sort_keys=True, default=str).encode()).hexdigest()
        np.save(out/'policy_state.npy', batch['observation.state'].detach().cpu().numpy())
        assert all(img.device.type == 'cuda' for img in images), '图像没有进入 GPU'
        assert next(self.parameters()).device.type == 'cuda', '模型没有进入 GPU'
        assert batch['observation.state'].shape[-1] == 8
        for key, image in zip(keys, images):
            a = image[0].detach().float().cpu().numpy()
            np.save(out/f'policy_tensor_{key.rsplit(".", 1)[-1]}.npy', a)
            rgb = np.rint(np.clip((a.transpose(1, 2, 0)+1)*127.5, 0, 255)).astype(np.uint8)
            Image.fromarray(rgb).save(out/f'policy_rgb_{key.rsplit(".", 1)[-1]}.png')
        torch.cuda.reset_peak_memory_stats()
    evidence['inference_calls'] += 1
    return images, masks


def observed_reset(self, *args, **kwargs):
    state_id = self.init_state_id % len(self._init_states)
    obs, info = original_reset(self, *args, **kwargs)
    if 'state_id' not in evidence:
        evidence.update(task_id=self.task_id, task_name=self.task, instruction=self.task_description,
                        state_id=state_id, horizon=self._max_episode_steps, control_mode=self.control_mode)
        np.save(out/'initial_state.npy', np.asarray(self._init_states[state_id]))
        for key, rgb in obs['pixels'].items():
            Image.fromarray(rgb).save(out/f'raw_reset_{key}.png')
        (out/'reset_robot_state.json').write_text(json.dumps(obs['robot_state'], default=lambda x: np.asarray(x).tolist(), indent=2))
    return obs, info


def observed_step(self, action):
    assert action.shape == (7,) and np.isfinite(action).all(), '非法真实动作'
    obs, reward, terminated, truncated, info = original_step(self, action)
    evidence['env_steps'] += 1
    success = bool(info['is_success'])
    with (out/'steps.jsonl').open('a') as f:
        f.write(json.dumps({'step': evidence['env_steps'], 'action': action.tolist(),
            'eef_pos': obs['robot_state']['eef']['pos'].tolist(), 'reward': float(reward),
            'success': success, 'terminated': bool(terminated), 'truncated': bool(truncated)})+'\n')
    if success:
        evidence.update(status='success', success=True)
    elif terminated:
        evidence.update(status='valid_task_failure', success=False)
    return obs, reward, terminated, truncated, info


SmolVLAPolicy.prepare_images = observed_images
LiberoEnv.reset = observed_reset
LiberoEnv.step = observed_step
exit_code = 0
try:
    main()
    assert evidence['inference_calls'] > 0 and evidence['env_steps'] > 0
    if evidence['success'] is None:
        assert evidence['env_steps'] == evidence['horizon'], '未完成完整 episode'
        evidence.update(status='task_timeout', success=False)
    evidence['error_type'] = None
except BaseException as exc:
    exit_code = 1
    evidence.update(status='infrastructure_failure', success=None, error_type=type(exc).__name__,
                    traceback=traceback.format_exc())
    traceback.print_exc()
finally:
    evidence['elapsed_time'] = time.monotonic()-started
    evidence['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    evidence['gpu_peak_reserved_bytes'] = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
    (out/'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(evidence, ensure_ascii=False, default=str))
sys.exit(exit_code)
