Part IV - Service
=================

The core (Parts I-III) and the user interface are separate processes.  This part
defines what connects them: a wire protocol (Section 1), the session that owns a
reconstruction (Section 2), the assembly of a model from a data folder or a run
and the jobs that advance it (Section 3), the server (Section 4), the client
(Section 5) and the command line (Section 6).  Only ``protocol`` and ``client``
are needed on the interface side; they import neither PyTorch nor the physics.

.. automodule:: temsim.service.protocol
   :members:

.. automodule:: temsim.service.session
   :members:

.. automodule:: temsim.service.assemble
   :members:

.. automodule:: temsim.service.jobs
   :members:

.. automodule:: temsim.service.server
   :members:

.. automodule:: temsim.service.client
   :members:

.. automodule:: temsim.service.cli
   :members:
