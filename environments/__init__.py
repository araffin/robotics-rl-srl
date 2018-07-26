from enum import Enum


class PlottingType(Enum):
    PLOT_2D = 1
    PLOT_3D = 2


class ThreadingType(Enum):
    PROCESS = 1
    THREADING = 2
    NONE = 3
