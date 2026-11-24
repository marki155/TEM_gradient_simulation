"""Part V, Section 14: the batch tool -- many images, one pipeline, overnight.

* :mod:`temsim_ml.batch.job`    -- the job file, inputs, step kinds, status;
* :mod:`temsim_ml.batch.runner` -- the process that works through a job without a window;
* :mod:`temsim_ml.batch.render` -- the pictures of the found atoms;
* :mod:`temsim_ml.batch.window` -- the window that puts a job together, starts and watches it.

``python -m temsim_ml.batch [JOBDIR]`` opens the window, ``python -m temsim_ml.batch run JOBDIR``
runs a job directly.
"""
