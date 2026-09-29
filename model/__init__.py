from .GLNO.layers import GLNONet
from .gridGLNO.layers import GLNO1D, GLNO2D
MODEL_DICT = {
    "GLNO": GLNONet,
    'GLNO1D': GLNO1D,
    'GLNO2D': GLNO2D,
}