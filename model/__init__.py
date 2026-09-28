# import diffusion_net.layers_dfn
# import FNO.layers_fno
from .GLNO.layers import GLNONet
from .GLNO.layers_unet import GLNONet as GLNONet_unet
from .GLNO.layers_dense import GLNONet as GLNONet_dense
from .GLNO.layers_dual import GLNONet as GLNONet_dual
from .GLNO.layers_unet_k import GLNONet as GLNONet_unet_k
from .GLNO.layers_lno import GLNONet as LNONet
from .GLNO.layers_multicentral import MultiCentralGLNO
from .GLNO.layers_geohead import GLNONet as GLNONet_geohead
from .GLNO.layers_2_0 import GLNONet as GLNONet_2_0
from .GLNO.layers_1_1 import GLNONet as GLNONet_1_1
from .GEO_FNO.layers import GEO_FNO
from .GKNO.layers import GKNO
from .Sp2GNO.layers import Sp2GNOAdapter
from .GINO.layer import GINOAdapter
from .gridGLNO.layers import GLNO1D, GLNO2D
from .gridLNO.layers import LNO1D, LNO2D
from .gridFNO.layers import FNO1D, FNO2D
from .gridWNO.layers import WNO1D, WNO2D
from .gridCNO.layer1D import CNO1D
from .gridCNO.layer2D import CNO2D
from .UNet.unet1d import UNet1d
from .MLP.mlp import MLP
from .Transolver.layers import Model as Transolver
from .AMG.grapher import Grapher as AMG
from .LSM.layers import Model as LSM
from .GNOT.gnot import GNOT
from .GLNO.layers_HPM import HPMNet
from .RNO.layers import GLNONet as RNONet
from .gridRNO.layers import LNO1D as RNO1D
MODEL_DICT = {
    # vertices
    "GLNO": GLNONet,
    "GLNO_unet": GLNONet_unet,
    "GLNO_dense": GLNONet_dense,
    "GLNO_dual": GLNONet_dual,
    "GLNO_unet_k": GLNONet_unet_k,
    'GLNO_geohead': GLNONet_geohead,
    'HPM': HPMNet,
    'GLNO_2_0': GLNONet_2_0,
    'GLNO_1_1': GLNONet_1_1,
    'RNO': RNONet,
    'LNO': LNONet,
    # 'MultiCentralGLNO': MultiCentralGLNO,
    "GEO_FNO": GEO_FNO,
    'GKNO': GKNO,
    'Sp2GNO': Sp2GNOAdapter,
    'GINO': GINOAdapter,
    'GNOT': GNOT,
    'LSM': LSM,
    'Transolver': Transolver,
    'AMG': AMG,

    'Unet': UNet1d,
    'MLP': MLP,
    # grid
    'GLNO1D': GLNO1D,
    'GLNO2D': GLNO2D,
    'LNO1D': LNO1D,
    'LNO2D': LNO2D,
    'FNO1D': FNO1D,
    'FNO2D': FNO2D,
    'WNO1D': WNO1D,
    'WNO2D': WNO2D,
    'CNO1D': CNO1D,
    'CNO2D': CNO2D,
    'RNO1D': RNO1D,
}