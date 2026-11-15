"""Mask geometry uses half-open pixel bboxes (x0, y0, x1, y1)."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Square:
    x0: int
    y0: int
    side: int

    @property
    def bbox(self):
        return self.x0, self.y0, self.x0+self.side, self.y0+self.side

    @property
    def area(self):
        return self.side**2

    def validate(self, width, height):
        if self.side < 0 or self.x0 < 0 or self.y0 < 0 or self.x0+self.side > width or self.y0+self.side > height:
            raise ValueError('Mask square is outside image bounds')

    def to_dict(self):
        return {'bbox': list(self.bbox), 'side': self.side, 'pixel_area': self.area}


def square_at(center, side, width, height):
    if not isinstance(side, int) or side < 0:
        raise ValueError('side must be a nonnegative integer')
    square = Square(math.floor(center[0]-side/2+0.5), math.floor(center[1]-side/2+0.5), side)
    square.validate(width, height)
    return square


def coverage(target, square):
    target = np.asarray(target, dtype=bool)
    if not target.any():
        raise ValueError('No visible target pixels')
    square.validate(target.shape[1], target.shape[0])
    x0,y0,x1,y1 = square.bbox
    return float(target[y0:y1,x0:x1].sum()/target.sum())


def minimum_target_square(target, fraction):
    target = np.asarray(target, dtype=bool)
    if target.ndim != 2 or not target.any() or not 0 < fraction <= 1:
        raise ValueError('Need a nonempty 2D target and 0 < fraction <= 1')
    ys,xs = np.nonzero(target)
    center = (float(xs.mean()+0.5), float(ys.mean()+0.5))
    h,w = target.shape
    for side in range(1,min(h,w)+1):
        try:
            square = square_at(center,side,w,h)
        except ValueError:
            continue
        if coverage(target,square) >= fraction:
            return square
    raise ValueError('Requested centroid-centered target coverage cannot fit inside image')


def paired_squares(target, fraction, background_anchor, background_allowed):
    target_square = minimum_target_square(target,fraction)
    h,w = target.shape
    background_square = square_at(background_anchor,target_square.side,w,h)
    x0,y0,x1,y1 = background_square.bbox
    if not np.asarray(background_allowed,dtype=bool)[y0:y1,x0:x1].all():
        raise ValueError('Background square covers a forbidden pixel')
    if coverage(target,background_square) != 0:
        raise ValueError('Background square covers target')
    return target_square,background_square


def mask_rgb(rgb, square):
    rgb=np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('Expected HWC uint8 RGB')
    square.validate(rgb.shape[1],rgb.shape[0])
    result=rgb.copy()
    x0,y0,x1,y1=square.bbox
    result[y0:y1,x0:x1]=128
    return result
