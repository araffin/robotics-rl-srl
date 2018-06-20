from enum import Enum


class SRLType(Enum):
    SRL = 1
    ENVIRONMENT = 2  # defined as anything from the environment (joints, ground_truth, ...)
