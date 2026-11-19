"""Records layer of the 0.8 design (design/0.8/SPEC.md Sec. 3.1, 3.4, 9.1): build records and the ledger.

LABEL ``[extension]``. ``records.py`` holds ``SpecifiedRecord``, ``RouteRecord`` and ``BuildRecord``, and
``ledger.py`` holds ``Ledger`` (decision 149). The package has no units or shapes of its own, and each module
states them. Importing the package loads no submodule. Not certified: a record documents what a build did and
certifies nothing about accuracy.
"""
