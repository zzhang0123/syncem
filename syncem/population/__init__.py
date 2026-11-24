"""Population declarations and pairing on the 0.8 lattice.

LABEL [extension].

Declaration shapes include scalar factors and arrays of nodes in each variable's
units. Product and joint declarations describe the assumed measure. They
have no certified discrepancy with a physical source.
"""

from .blocks import FreeMoments, NonNegativeCells, Samples
from .ray import Ray
from .declaration import Population
from .pairing import pair

__all__ = ["Population", "pair", "FreeMoments", "NonNegativeCells", "Samples", "Ray"]
