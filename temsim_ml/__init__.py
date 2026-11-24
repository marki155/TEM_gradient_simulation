"""Part V, Section 13: the atom detector as stand-alone tools (predict, train).

These windows run the network themselves (torch), so they are a separate
application next to the core -- the main interface ``temsim_gui`` never imports
them, it starts them as processes and exchanges files with them:

* ``python -m temsim_ml.predict [--target F --i-target F] [--handoff OUT.npz]``
* ``python -m temsim_ml.train [--run DIR]``
* ``python -m temsim_ml.batch [JOBDIR]`` -- the batch tool: many images and stacks through one
  pipeline, overnight (:mod:`temsim_ml.batch`); ``python -m temsim_ml.batch run JOBDIR`` without a window.
"""
