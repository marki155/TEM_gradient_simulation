# Legacy core (archived)

The simulation core and interface that preceded the `temsim` / `temsim_gui` packages.
Nothing here is used by the new code.  It is kept for one purpose: the tests in
`tests/` compare the new physics and learning numerically against these modules
(the `legacy` fixture in `tests/conftest.py` puts this directory on `sys.path`).

The old scripts (`mains/`, `test_*.py`, `gui_qt.py`, ...) were written to run from the
repository root; their relative data paths no longer resolve from here and have to be adjusted
before they are used again.
