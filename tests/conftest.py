from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "colfov_b8s3.pt"


@pytest.fixture(scope="session")
def loaded():
    from colfov import load_model
    return load_model(WEIGHTS)


@pytest.fixture
def synthetic_frame():
    """A deterministic stand-in for an endoscopic frame: bright elliptical field on
    black, with a smaller bright inset. No clinical image is required by CI."""
    def make(h=720, w=1280, inset=None, seed=0):
        rng = np.random.default_rng(seed)
        f = np.zeros((h, w, 3), np.uint8)
        yy, xx = np.mgrid[0:h, 0:w]
        ell = ((xx - w / 2) / (w * 0.36)) ** 2 + ((yy - h / 2) / (h * 0.46)) ** 2 <= 1
        base = np.array([60, 90, 180], np.float32)
        f[ell] = np.clip(base * (0.8 + 0.4 * rng.random((int(ell.sum()), 1))), 0, 255)
        if inset:
            x, y, iw, ih = inset
            f[y:y + ih, x:x + iw] = np.clip(base * 1.15, 0, 255).astype(np.uint8)
        return f
    return make
