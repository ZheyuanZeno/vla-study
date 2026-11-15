"""Synthetic fixture used only to test export of the requested scientific figures."""
import numpy as np
from PIL import Image
from src.analysis.bootstrap import summarize
from src.analysis.run import plots


def test_synthetic_figure_exports_are_readable(tmp_path):
    matrix=np.array([[1,1,0,0,1,1,1],[0,0,0,0,0,0,1]],dtype=float)
    report=summarize({f'synthetic:{i}':matrix for i in range(4)},5000,1)
    plots(report,tmp_path)
    paths=list(tmp_path.glob('*.png'))
    assert len(paths)==4
    for path in paths:
        with Image.open(path) as image:
            assert image.width>500 and image.height>300
            image.verify()
