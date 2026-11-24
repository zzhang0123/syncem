"""Distribution and family declarations (SPEC Sec. 3.5).

LABEL [extension].

Each law declares its units and the shape of its quadrature. A finite
quadrature carries no certified integration error by default.
"""

from ._common import Distribution as Distribution
from .azimuth import AzimuthDensity as AzimuthDensity
from .azimuth import AzimuthFactor as AzimuthFactor
from .azimuth import UniformAzimuth as UniformAzimuth
from .densities import Delta as Delta
from .densities import Isotropic as Isotropic
from .densities import LogNormal as LogNormal
from .densities import Maxwellian as Maxwellian
from .densities import Normal as Normal
from .densities import Tabulated as Tabulated
from .densities import Uniform as Uniform
from .densities import UniformAngle as UniformAngle
from .aged import AgedPowerLaw as AgedPowerLaw
from .electrons import BrokenPowerLaw as BrokenPowerLaw
from .electrons import PowerLaw as PowerLaw
from .electrons import PowerLawMixture as PowerLawMixture
from .parameters import Conditional as Conditional
from .parameters import Free as Free
from .screens import CumulantScreen as CumulantScreen
from .screens import EmpiricalScreen as EmpiricalScreen
from .screens import GammaScreen as GammaScreen
from .screens import GaussianScreen as GaussianScreen
from .screens import LaplaceScreen as LaplaceScreen
from .screens import NoRotation as NoRotation
from .user import UserFamily as UserFamily
from .user import family as family
