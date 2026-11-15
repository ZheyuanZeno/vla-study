"""系统、CUDA、依赖及真实 LIBERO/EGL reset/step 验证。异常不会作为成功。"""
import argparse
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulator', action='store_true')
    parser.add_argument('--output', default='artifacts/environment.json')
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'python': sys.version, 'executable': sys.executable,
              'platform': platform.platform(), 'mujoco_gl': os.getenv('MUJOCO_GL'),
              'checks': {}, 'status': 'failed'}
    for command in [['nvidia-smi'], ['df', '-h', '.'], ['free', '-h'], ['uname', '-a']]:
        p = subprocess.run(command, capture_output=True, text=True)
        report['checks'][command[0]] = {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    try:
        report['versions'] = {}
        for distribution, module in [('torch','torch'), ('torchvision','torchvision'),
              ('lerobot','lerobot'), ('transformers','transformers'),
              ('mujoco','mujoco'), ('robosuite','robosuite'), ('hf-libero','libero')]:
            importlib.import_module(module)
            report['versions'][distribution] = metadata.version(distribution)
        import torch
        assert torch.cuda.is_available(), 'CUDA 不可用'
        x = torch.eye(32, device='cuda')
        assert torch.equal(x @ x, x)
        torch.cuda.synchronize()
        report['cuda'] = {'runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
                          'memory': list(torch.cuda.mem_get_info()), 'compute_verified': True}
        if args.simulator:
            import numpy as np
            from PIL import Image
            from lerobot.envs.libero import LiberoEnv, _get_suite
            env = LiberoEnv(_get_suite('libero_spatial'), 0, 'libero_spatial', obs_type='pixels_agent_pos')
            try:
                obs, info = env.reset(seed=42)
                images = {}
                for camera, rgb in obs['pixels'].items():
                    assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
                    assert rgb.std() > 1, '无有效图像'
                    Image.fromarray(rgb).save(output.parent / f'reset_{camera}.png')
                    images[camera] = {'shape': list(rgb.shape), 'std': float(rgb.std())}
                action = np.array([0., 0., 0., 0., 0., 0., -1.])
                _, reward, terminated, truncated, step_info = env.step(action)
                assert isinstance(step_info['is_success'], (bool, np.bool_))
                states = np.asarray(env._init_states)
                report['simulator'] = {'task': env.task, 'instruction': env.task_description,
                    'images': images, 'init_states': len(states),
                    'unique_init_states': len(np.unique(states, axis=0)),
                    'state_shape': list(states.shape), 'reset_state_id': 0,
                    'reward': float(reward), 'success': bool(step_info['is_success']),
                    'real_reset_and_step': True}
            finally:
                env.close()
        report['status'] = 'passed'
    except Exception:
        report['error'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
