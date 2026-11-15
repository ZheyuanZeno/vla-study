"""A scoped hook at the pinned upstream resize function, before SigLIP normalization.

Only single-process, synchronous inference is supported. The upstream method is
called unchanged; its camera order is checked on every call. No library file is edited.
"""
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from lerobot.policies.smolvla import modeling_smolvla as upstream


@contextmanager
def image_intervention(policy, square, output_dir):
    original_prepare=policy.prepare_images
    original_resize=upstream.resize_with_pad
    count=0
    output_dir.mkdir(parents=True,exist_ok=True)
    if tuple(policy.config.resize_imgs_with_padding or ()) != (512,512):
        raise ValueError('This experiment is frozen to the checkpoint 512x512 resize')
    if policy.config.compile_model:
        raise ValueError('Scoped image hook requires uncompiled synchronous inference')

    def prepare(batch):
        nonlocal count
        keys=[key for key in policy.config.image_features if key in batch]
        assert keys == ['observation.images.image','observation.images.image2'], keys
        call_index=0
        current_square=square() if callable(square) else square
        def resize(image,*args,**kwargs):
            nonlocal call_index
            key=keys[call_index]
            call_index+=1
            resized=original_resize(image,*args,**kwargs)
            if key == 'observation.images.image':
                assert resized.shape == (1,3,512,512)
                changed=resized.clone()
                if current_square is not None:
                    current_square.validate(512,512)
                    x0,y0,x1,y1=current_square.bbox
                    changed[:,:,y0:y1,x0:x1]=128.0/255.0
                if count == 0:
                    for name,value in [('original',resized),('masked',changed)]:
                        a=value[0].detach().float().cpu().permute(1,2,0).numpy()
                        np.save(output_dir/f'{name}_rgb_float.npy',a)
                        Image.fromarray(np.rint(np.clip(a,0,1)*255).astype(np.uint8)).save(output_dir/f'{name}.png')
                return changed
            return resized
        with patch.object(upstream,'resize_with_pad',resize):
            images,masks=original_prepare(batch)
        assert call_index == 2
        if current_square is not None:
            x0,y0,x1,y1=current_square.bbox
            assert torch.allclose(images[0][:,:,y0:y1,x0:x1],torch.full_like(images[0][:,:,y0:y1,x0:x1],128.0/255.0)*2.0-1.0)
        count+=1
        return images,masks

    policy.prepare_images=prepare
    try:
        yield lambda: count
    finally:
        policy.prepare_images=original_prepare
