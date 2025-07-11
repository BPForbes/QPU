import unittest
import numpy as np
import os
from itertools import product
from qpu.ast import parse_command
from qpu.qpu_base import QuantumProcessorUnit
from cli import CircuitSimulator

def run_process(proc_file, proc_name, params):
    sim = CircuitSimulator(QuantumProcessorUnit(num_qubits=5))
    sim.current_cycle = 0
    cwd = os.getcwd()
    proto_dir = os.path.dirname(proc_file)
    os.chdir(proto_dir)
    try:
        # Compile the subprocess
        compile_cmd = f"COMPILEPROCESS --NAME {proc_name} {os.path.basename(proc_file)}"
        parse_command(compile_cmd).evaluate(
            sim.memory, sim.current_cycle, sim.qpu, sim.hilbert, sim
        )
        # Call it with the given parameters
        call_cmd = f"CALL {proc_name} -I {' '.join(params)}"
        parse_command(call_cmd).evaluate(
            sim.memory, sim.current_cycle, sim.qpu, sim.hilbert, sim
        )
    finally:
        os.chdir(cwd)
    return sim

# -------------------------------------------------------------------
# Single‐Bit Full Adder Tests (one test method per input combo)
# -------------------------------------------------------------------
class TestSingleBitAdder(unittest.TestCase):
    zero = np.array([1.0, 0.0], dtype=complex)
    one  = np.array([0.0, 1.0], dtype=complex)
    proc_file = os.path.join(os.path.dirname(__file__), "SingleBitFullAdder.txt")
    proc_name = "SingleBitFullAdder"

# All combinations of A, B, Cin
_single_bit_combos = [
    ("0p","0p","0p", TestSingleBitAdder.zero, TestSingleBitAdder.zero),
    ("0p","0p","1p", TestSingleBitAdder.one,  TestSingleBitAdder.zero),
    ("0p","1p","0p", TestSingleBitAdder.one,  TestSingleBitAdder.zero),
    ("0p","1p","1p", TestSingleBitAdder.zero, TestSingleBitAdder.one),
    ("1p","0p","0p", TestSingleBitAdder.one,  TestSingleBitAdder.zero),
    ("1p","0p","1p", TestSingleBitAdder.zero, TestSingleBitAdder.one),
    ("1p","1p","0p", TestSingleBitAdder.zero, TestSingleBitAdder.one),
    ("1p","1p","1p", TestSingleBitAdder.one,  TestSingleBitAdder.one),
]

def _make_single_bit_test(A, B, Cin, exp_sum, exp_cout):
    def test(self):
        sim = run_process(
            TestSingleBitAdder.proc_file,
            TestSingleBitAdder.proc_name,
            [A, B, Cin],
        )
        # find the latest cycles for Sum (3) and Cout (4)
        s_cycle = max(sim.memory[3].keys())
        c_cycle = max(sim.memory[4].keys())
        s = sim.memory[3][s_cycle]
        c = sim.memory[4][c_cycle]
        np.testing.assert_allclose(s, exp_sum, atol=1e-7)
        np.testing.assert_allclose(c, exp_cout, atol=1e-7)
    return test

# Dynamically bind one test method per combination
for idx, combo in enumerate(_single_bit_combos):
    name = f"test_single_bit_{idx}_{combo[0]}_{combo[1]}_{combo[2]}"
    setattr(TestSingleBitAdder, name, _make_single_bit_test(*combo))


# -------------------------------------------------------------------
# Two‐Bit Full Adder Tests (one test method per input combo)
# -------------------------------------------------------------------
class TestTwoBitAdder(unittest.TestCase):
    zero = np.array([1.0, 0.0], dtype=complex)
    one  = np.array([0.0, 1.0], dtype=complex)
    proc_file = os.path.join(os.path.dirname(__file__), "TwoBitFullAdder.txt")
    proc_name = "TwoBitFullAdder"

    @staticmethod
    def expected(A0, A1, B0, B1, Cin):
        # Convert "0p"/"1p" to bits
        a = (int(A1[0]) << 1) | int(A0[0])
        b = (int(B1[0]) << 1) | int(B0[0])
        cin = int(Cin[0])
        total = a + b + cin
        s0   = (total >> 0) & 1
        s1   = (total >> 1) & 1
        cout = (total >> 2) & 1
        return (
            TestTwoBitAdder.one  if s0 else TestTwoBitAdder.zero,
            TestTwoBitAdder.one  if s1 else TestTwoBitAdder.zero,
            TestTwoBitAdder.one  if cout else TestTwoBitAdder.zero,
        )

# Generate all combos of A0, A1, B0, B1, Cin
_bits = ["0p", "1p"]
_two_bit_combos = [
    (A0, A1, B0, B1, Cin, *TestTwoBitAdder.expected(A0, A1, B0, B1, Cin))
    for A0, A1, B0, B1, Cin in product(_bits, repeat=5)
]

def _make_two_bit_test(A0, A1, B0, B1, Cin, exp_s0, exp_s1, exp_cout):
    def test(self):
        sim = run_process(
            TestTwoBitAdder.proc_file,
            TestTwoBitAdder.proc_name,
            [A0, A1, B0, B1, Cin],
        )
        s0_cycle = max(sim.memory["Sum0"].keys())
        s1_cycle = max(sim.memory["Sum1"].keys())
        cout_cycle = max(sim.memory["Cout"].keys())
        s0 = sim.memory["Sum0"][s0_cycle]
        s1 = sim.memory["Sum1"][s1_cycle]
        cout = sim.memory["Cout"][cout_cycle]
        np.testing.assert_allclose(s0, exp_s0, atol=1e-7)
        np.testing.assert_allclose(s1, exp_s1, atol=1e-7)
        np.testing.assert_allclose(cout, exp_cout, atol=1e-7)
    return test

# Bind each two-bit combination as its own test method
for idx, combo in enumerate(_two_bit_combos):
    A0, A1, B0, B1, Cin, *_ = combo
    name = f"test_two_bit_{idx}_{A0}{A1}_{B0}{B1}_{Cin}"
    setattr(TestTwoBitAdder, name, _make_two_bit_test(*combo))


# -------------------------------------------------------------------
# Direct gate interaction with Hilbert space
# -------------------------------------------------------------------
class TestGateHilbert(unittest.TestCase):
    def test_cnot_logs_hilbert(self):
        qpu = QuantumProcessorUnit(num_qubits=2)
        hilbert = qpu.hilbert if hasattr(qpu, 'hilbert') else None
        if hilbert is None:
            from qpu.hilbert import HilbertSpace
            hilbert = HilbertSpace()
        # initialize control |1> and target |0>
        qpu.local_states[0] = np.array([0,1], dtype=complex)
        qpu.local_states[1] = np.array([1,0], dtype=complex)
        qpu.rebuild_global_state()

        qpu.apply_cnot(0, 1, cycle=0, hilbert=hilbert)

        self.assertIn((1, 0), hilbert.space)
        np.testing.assert_allclose(hilbert.space[(1,0)], np.array([0,1], dtype=complex))


if __name__ == "__main__":
    unittest.main()
