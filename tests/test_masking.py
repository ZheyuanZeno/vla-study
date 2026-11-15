import numpy as np
import pytest
from src.masking.geometry import Square, coverage, mask_rgb, minimum_target_square, paired_squares


def test_zero_mask_identity():
    rgb=np.random.default_rng(1).integers(0,256,(40,50,3),dtype=np.uint8)
    assert np.array_equal(mask_rgb(rgb,Square(3,4,0)),rgb)


@pytest.mark.parametrize('fraction',[.25,.5,.75])
def test_minimal_centered_square_and_matched_background(fraction):
    target=np.zeros((100,150),dtype=bool); target[35:55,25:45]=True
    allowed=np.ones_like(target); allowed[target]=False
    t,b=paired_squares(target,fraction,(110,70),allowed)
    assert t.side==b.side and t.area==b.area
    assert coverage(target,t)>=fraction and coverage(target,b)==0
    from src.masking.geometry import square_at
    for smaller in range(1,t.side):
        assert coverage(target,square_at((35,45),smaller,150,100)) < fraction
    rgb=np.zeros((100,150,3),dtype=np.uint8)
    result=mask_rgb(rgb,t); x0,y0,x1,y1=t.bbox
    assert (result[y0:y1,x0:x1]==128).all()
    assert np.count_nonzero(np.any(result!=rgb,axis=-1))==t.area


def test_invalid_masks_fail():
    with pytest.raises(ValueError): minimum_target_square(np.zeros((10,10)),.25)
    with pytest.raises(ValueError): mask_rgb(np.zeros((10,10,3),dtype=np.uint8),Square(9,9,2))
    target=np.ones((20,20),dtype=bool)
    with pytest.raises(ValueError): paired_squares(target,.25,(2,2),np.zeros_like(target))
