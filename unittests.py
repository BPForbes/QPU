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
        compile_cmd = f"COMPILEPROCESS --NAME {proc_name} {os.path.basename(proc_file)}"
        parse_command(compile_cmd).evaluate(
            sim.memory, sim.current_cycle, sim.qpu, sim.hilbert, sim
        )
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
        s_cycle = max(sim.memory[3].keys())
        c_cycle = max(sim.memory[4].keys())
        s = sim.memory[3][s_cycle]
        c = sim.memory[4][c_cycle]
        np.testing.assert_allclose(s, exp_sum, atol=1e-7)
        np.testing.assert_allclose(c, exp_cout, atol=1e-7)
    return test

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

for idx, combo in enumerate(_two_bit_combos):
    A0, A1, B0, B1, Cin, *_ = combo
    name = f"test_two_bit_{idx}_{A0}{A1}_{B0}{B1}_{Cin}"
    setattr(TestTwoBitAdder, name, _make_two_bit_test(*combo))


# -------------------------------------------------------------------
# Four‐Bit Full Adder Tests (one test method per input combo)
# -------------------------------------------------------------------
class TestFourBitAdder(unittest.TestCase):
    zero = np.array([1.0, 0.0], dtype=complex)
    one  = np.array([0.0, 1.0], dtype=complex)
    proc_file = os.path.join(os.path.dirname(__file__), "FourBitFullAdder.txt")
    proc_name = "FourBitFullAdder"

    @staticmethod
    def expected(A0, A1, A2, A3, B0, B1, B2, B3, Cin):
        a = (int(A3[0]) << 3) | (int(A2[0]) << 2) | (int(A1[0]) << 1) | int(A0[0])
        b = (int(B3[0]) << 3) | (int(B2[0]) << 2) | (int(B1[0]) << 1) | int(B0[0])
        cin = int(Cin[0])
        total = a + b + cin
        s0   = (total >> 0) & 1
        s1   = (total >> 1) & 1
        s2   = (total >> 2) & 1
        s3   = (total >> 3) & 1
        cout = (total >> 4) & 1
        return (
            TestFourBitAdder.one  if s0 else TestFourBitAdder.zero,
            TestFourBitAdder.one  if s1 else TestFourBitAdder.zero,
            TestFourBitAdder.one  if s2 else TestFourBitAdder.zero,
            TestFourBitAdder.one  if s3 else TestFourBitAdder.zero,
            TestFourBitAdder.one  if cout else TestFourBitAdder.zero,
        )

_four_bit_combos = [
    (A0, A1, A2, A3, B0, B1, B2, B3, Cin, *TestFourBitAdder.expected(A0, A1, A2, A3, B0, B1, B2, B3, Cin))
    for A0, A1, A2, A3, B0, B1, B2, B3, Cin in product(_bits, repeat=9)
]

def _make_four_bit_test(A0, A1, A2, A3, B0, B1, B2, B3, Cin, exp_s0, exp_s1, exp_s2, exp_s3, exp_cout):
    def test(self):
        sim = run_process(
            TestFourBitAdder.proc_file,
            TestFourBitAdder.proc_name,
            [A0, A1, A2, A3, B0, B1, B2, B3, Cin],
        )
        s0_cycle = max(sim.memory["Sum0"].keys())
        s1_cycle = max(sim.memory["Sum1"].keys())
        s2_cycle = max(sim.memory["Sum2"].keys())
        s3_cycle = max(sim.memory["Sum3"].keys())
        cout_cycle = max(sim.memory["C4"].keys())
        s0 = sim.memory["Sum0"][s0_cycle]
        s1 = sim.memory["Sum1"][s1_cycle]
        s2 = sim.memory["Sum2"][s2_cycle]
        s3 = sim.memory["Sum3"][s3_cycle]
        cout = sim.memory["C4"][cout_cycle]
        np.testing.assert_allclose(s0, exp_s0, atol=1e-7)
        np.testing.assert_allclose(s1, exp_s1, atol=1e-7)
        np.testing.assert_allclose(s2, exp_s2, atol=1e-7)
        np.testing.assert_allclose(s3, exp_s3, atol=1e-7)
        np.testing.assert_allclose(cout, exp_cout, atol=1e-7)
    return test

for idx, combo in enumerate(_four_bit_combos):
    name = f"test_four_bit_{idx}_" + "".join(combo[:9])
    setattr(TestFourBitAdder, name, _make_four_bit_test(*combo))


# -------------------------------------------------------------------
# Direct gate interaction with Hilbert space
# -------------------------------------------------------------------
class TestGateHilbert(unittest.TestCase):
    def test_cnot_logs_hilbert(self):
        qpu = QuantumProcessorUnit(num_qubits=2)
        hilbert = getattr(qpu, 'hilbert', None) or __import__('qpu.hilbert').hilbert.HilbertSpace()
        qpu.local_states[0] = np.array([0,1], dtype=complex)
        qpu.local_states[1] = np.array([1,0], dtype=complex)
        qpu.rebuild_global_state()

        qpu.apply_cnot(0, 1, cycle=0, hilbert=hilbert)

        self.assertIn((1, 0), hilbert.space)
        np.testing.assert_allclose(hilbert.space[(1,0)], np.array([0,1], dtype=complex))


# -------------------------------------------------------------------
# Cycle mechanics & state snapshot tests
# -------------------------------------------------------------------
class TestCycleMechanics(unittest.TestCase):
    def test_increasecycle(self):
        sim = CircuitSimulator(QuantumProcessorUnit(num_qubits=2))
        start = sim.current_cycle
        parse_command("INCREASECYCLE").evaluate(sim.memory, sim.current_cycle, sim.qpu, sim.hilbert, sim)
        self.assertEqual(sim.current_cycle, start + 1)

    def test_save_and_load_state(self):
        sim = CircuitSimulator(QuantumProcessorUnit(num_qubits=1))
        parse_command("SET 0 1p").evaluate(sim.memory, sim.current_cycle, sim.qpu, sim.hilbert, sim)
        parse_command("SAVE_STATE snap").evaluate(sim.memory, sim.current_cycle, sim.qpu, sim.hilbert, sim)
        parse_command("SET 0 0p").evaluate(sim.memory, sim.current_cycle, sim.qpu, sim.hilbert, sim)
        parse_command("LOAD_STATE snap").evaluate(sim.memory, sim.current_cycle, sim.qpu, sim.hilbert, sim)
        np.testing.assert_allclose(sim.qpu.local_states[0], np.array([0.0,1.0], dtype=complex))


if __name__ == "__main__":
    unittest.main()
