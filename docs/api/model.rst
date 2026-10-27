syncem.model
============

Finite joint response of channel Stokes spectra, error budget and declared assumptions.

syncem.model.index
------------------

.. automodule:: syncem.model.index

syncem.model.errors
-------------------

.. automodule:: syncem.model.errors

syncem.model.channels
---------------------

.. automodule:: syncem.model.channels

syncem.model.phase
------------------

.. automodule:: syncem.model.phase

syncem.model.screens
--------------------

.. automodule:: syncem.model.screens

syncem.model.moments
--------------------

.. automodule:: syncem.model.moments

syncem.model.bounds
-------------------

.. automodule:: syncem.model.bounds

syncem.model.assumptions
------------------------

``declare``, ``Assume`` and ``ASSUMPTIONS`` back the ``assumptions=`` keyword of
``fit_linear``, ``fit_combinations``, ``LogDensity``,
``JointMoments.from_samples`` and ``SpectralBasis.constrained``. ``angles_independent`` and ``known_angles`` are the
uncoupled case of ``eq: uncoupled moments``, with unknown (bilinear) and known
(affine) coefficients ``a_lk``; Section 7.6 of the design document has the rules.

.. automodule:: syncem.model.assumptions

syncem.model.kernels
--------------------

.. automodule:: syncem.model.kernels

syncem.model.harmonic
---------------------

.. automodule:: syncem.model.harmonic

syncem.model.high_order_harmonic
--------------------------------

.. automodule:: syncem.model.high_order_harmonic

syncem.model.high_order_basis
-----------------------------

.. automodule:: syncem.model.high_order_basis

syncem.model.basis
------------------

.. automodule:: syncem.model.basis

syncem.model.predict
--------------------

.. automodule:: syncem.model.predict

syncem.model.adapters
---------------------

.. automodule:: syncem.model.adapters
