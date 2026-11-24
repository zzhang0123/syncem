Spectral bases and moment inference
===================================

Public entry points
-------------------

.. autofunction:: syncem.plan
.. autofunction:: syncem.spectral_basis
.. autofunction:: syncem.predict
.. autofunction:: syncem.fit
.. autofunction:: syncem.problem
.. autofunction:: syncem.compare
.. autofunction:: syncem.paper.population
.. autofunction:: syncem.paper.regime_basis
.. autofunction:: syncem.paper.energy_basis

Inputs and representations
--------------------------

.. autoclass:: syncem.Channels
.. autoclass:: syncem.Cells
.. autoclass:: syncem.Lattice
.. autoclass:: syncem.Legendre
.. autoclass:: syncem.AzimuthHarmonic
.. autoclass:: syncem.AzimuthFourier
.. autoclass:: syncem.Population
.. autoclass:: syncem.Conditional
.. autoclass:: syncem.CumulantScreen
.. autoclass:: syncem.FreeMoments
.. autoclass:: syncem.NonNegativeCells
.. autoclass:: syncem.StokesData

Bases and results
-----------------

.. autoclass:: syncem.SpectralBasis
   :members: at, build, design, direct, fit, load, moments, predict, save, select, with_order
.. autoclass:: syncem.BasisSum
.. autoclass:: syncem.RegimeBasis
   :members: build, fit, fit_family, moments, predict, problem
.. autoclass:: syncem.HarmonicShapeBasis
   :members: build, moments, predict, at, spacing, z_edges
   :exclude-members: __init__, __new__
.. autoclass:: syncem.HarmonicShapeStatistics
   :members: save, load
   :exclude-members: __init__, __new__
.. autoclass:: syncem.Spectrum
.. autoclass:: syncem.Ledger
.. autoclass:: syncem.ErrorTerm
   :no-index:
.. autoclass:: syncem.fitting.problem.Problem
   :members: constrain, log_prob, log_likelihood, profiled, solve, unconstrain
