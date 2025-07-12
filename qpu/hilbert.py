"""
hilbert.py

Implements the Hilbert space storage system.
Values are stored using keys as (register, cycle).
The view() method produces a nicely formatted, ket–formatted output.
"""

from qpu.qpu_base import format_qubit_state

class HilbertSpace:
    def __init__(self):
        """
        Initializes an empty Hilbert space.
        """
        self.space = {}

    def copy(self):
        """
        Returns a shallow copy of the HilbertSpace.
        """
        new_hilbert = HilbertSpace()
        new_hilbert.space = self.space.copy()
        return new_hilbert

    def output(self, register, cycle: int, value):
        """
        Stores the given value under key (register, cycle).
        """
        self.space[(register, cycle)] = value

    def output_all(self, cycle: int, value):
        """Record a full-system snapshot under key '__all__'."""
        self.space[("__all__", cycle)] = value

    def input(self, register, cycle: int):
        """
        Retrieves and removes the value stored under (register, cycle).
        """
        return self.space.pop((register, cycle), None)

    def prune(self, max_cycle: int):
        """
        Removes entries with cycle numbers greater than max_cycle.
        """
        self.space = {k: v for k, v in self.space.items() if k[1] <= max_cycle}

    def view(self) -> str:
        """
        Returns a formatted string of the Hilbert space.
        Each register is shown with its cycle and ket–formatted state.
        """
        if not self.space:
            return "Hilbert Space is empty."
        lines = []
        # Sort keys by cycle then register
        sorted_keys = sorted(self.space.keys(), key=lambda x: (x[1], str(x[0])))
        for key, cycle in sorted_keys:
            state = self.space[(key, cycle)]
            lines.append(f"Register {key} @ Cycle {cycle}: {format_qubit_state(state)}")
        return "\n".join(lines)
