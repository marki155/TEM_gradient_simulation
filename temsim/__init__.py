"""temsim: differentiable (S)TEM image simulation and structure fitting.

The package is organised like a paper.  Each part opens with its theory, and the
code below the theory implements it, referring to the numbered equations.

========  ==================  ===================================================
Part      Package             Question answered
========  ==================  ===================================================
I         ``temsim.physics``  What image does a given structure produce?
II        ``temsim.learning`` Which structure and parameters explain an image?
III       ``temsim.data``     How are models, settings and data described,
                              stored and migrated?
IV        ``temsim.service``  How does the core talk to a user interface?
========  ==================  ===================================================

Dependencies point one way only::

    data  <-  physics  <-  learning  <-  service  <-  (process boundary)  <-  temsim_gui

``temsim.data`` needs only numpy; the GUI process imports ``temsim.data.schema``
and ``temsim.service.protocol`` and never torch.
"""

__version__ = "0.1.0"
