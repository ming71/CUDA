from .ARF import MappingRotate
from .RIE import ORAlign1d

def mapping_rotate(input, indices):
    return MappingRotate()(input, indices)

def oraligned1d(input, nOrientation):
    return ORAlign1d()(input, nOrientation)

__all__ = ["oralign1d", "mapping_rotate"]