from .glno_dataset import *
from .glno_dataset__ import *
from .grid_dataset import *
# from .prepare_data import prepare_data

DATASET_DICT={
    "poisson": PoissonBase,
    "poissonunstruc": PoissonUnstrucDataset,
    "car": CarCFD,
    'airfoil': AIRFOILBase,
    'cylinder_flow': CylinderFlow,
    "rna": RNAMeshDataset,
    'human': HumanMeshDataset,
    'pendulum': Grid1DDataset,
    'lorenz': Grid1DDataset,
    'duffing': Grid1DDataset,
    'beam': Grid2DDataset,
    'diffusion': Grid2DDataset,
    'reacdiffusion': Grid2DDataset,
    'shrec11_simplified': ShrecMeshDataset,
    'cortex': CortexDataset,
    'intra': IntraDataset,
    'turbulent': WellDataset,
}