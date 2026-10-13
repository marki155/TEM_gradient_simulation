"""Part V: the graphical client of the TEM gradient simulation.

The interface is a client of the core (Part IV).  It imports neither the physics nor the
learning part, and it does not use torch; it sees the model only as frames, light events
and the answers to commands.
"""
