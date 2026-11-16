"""Decode every fresh Pilot video and compare its frame count with real steps."""
import json
from pathlib import Path

import imageio.v2 as imageio


def main():
    root = Path('artifacts/pilot_fresh_v1')
    evidence = []
    for task_id in range(4):
        rows = [json.loads(line) for line in
                (root / f'libero_spatial_{task_id}' / 'results.jsonl').read_text().splitlines()]
        assert len(rows) == 7 and all(row['status'] != 'infrastructure_failure' for row in rows)
        for row in rows:
            path = Path(row['attempt_dir']) / 'rollout.mp4'
            with imageio.get_reader(path) as reader:
                frames = sum(1 for _ in reader)
            assert frames == row['env_steps'] + 1, (path, frames, row['env_steps'])
            evidence.append({'file': str(path), 'decoded_frames': frames,
                             'env_steps': row['env_steps'], 'passed': True})
    report = {'real_videos': len(evidence), 'all_passed': True, 'videos': evidence}
    (root / 'video_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'real_videos': len(evidence), 'all_passed': True}))


if __name__ == '__main__':
    main()
