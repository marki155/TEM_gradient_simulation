Part V - Graphical client
=========================

The interface is a client of the core.  It imports neither the physics nor the
learning part and does not use PyTorch: everything it shows arrives as frames
(Section 1), and everything it does is a command of the protocol (Part IV).
Sections 1-3 need no display (frame model, tiles, marks); Section 4 draws them;
Section 5 connects Qt to the core; Sections 6-9 are the panels; Section 10 is the
controller that translates between panels and commands; Sections 11 and 12 compose
the window and start it.

.. automodule:: temsim_gui.frame_model
   :members:

.. automodule:: temsim_gui.tiles
   :members:

.. automodule:: temsim_gui.overlays
   :members:

.. automodule:: temsim_gui.canvas
   :members:

.. automodule:: temsim_gui.connection
   :members:

.. automodule:: temsim_gui.widgets
   :members:

.. automodule:: temsim_gui.values_editor
   :members:

.. automodule:: temsim_gui.open_dialog
   :members:

.. automodule:: temsim_gui.train_editor
   :members:

.. automodule:: temsim_gui.panels
   :members:

.. automodule:: temsim_gui.controller
   :members:

.. automodule:: temsim_gui.main_window
   :members:

.. automodule:: temsim_gui.launch
   :members:
