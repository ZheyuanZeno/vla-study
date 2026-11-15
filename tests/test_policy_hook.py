from types import SimpleNamespace
import numpy as np
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from src.masking.geometry import Square
from src.masking.policy_hook import image_intervention
from src.masking.segmentation import decode_id_rgb


def test_numpy2_id_decoding_does_not_overflow():
    rgb=np.array([[[255,255,255],[0,1,0],[0,0,1]]],dtype=np.uint8)
    assert decode_id_rgb(rgb).tolist()==[[16777215,256,65536]]


def test_actual_upstream_hook_position_and_wrist_identity(tmp_path):
    config=SimpleNamespace(image_features={'observation.images.image':None,'observation.images.image2':None},
                           resize_imgs_with_padding=(512,512),compile_model=False,empty_cameras=0)
    policy=SimpleNamespace(config=config)
    policy.prepare_images=lambda b: SmolVLAPolicy.prepare_images(policy,b)
    batch={key:torch.rand((1,3,256,256)) for key in config.image_features}
    baseline,_=policy.prepare_images(batch)
    square=Square(100,200,50)
    for selected in [None,Square(0,0,0),square]:
        with image_intervention(policy,selected,tmp_path) as calls:
            for _ in range(2):
                changed,_=policy.prepare_images(batch)
                assert torch.equal(changed[1],baseline[1])
                if selected is None or selected.area==0:
                    assert torch.equal(changed[0],baseline[0])
                else:
                    outside=torch.ones((512,512),dtype=torch.bool); outside[200:250,100:150]=False
                    assert torch.equal(changed[0][:,:,outside],baseline[0][:,:,outside])
                    assert torch.allclose(changed[0][:,:,200:250,100:150],torch.full((1,3,50,50),128/255)*2-1)
                assert batch['observation.images.image'].shape == (1,3,256,256)
            assert calls()==2
