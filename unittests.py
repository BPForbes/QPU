import unittest
import numpy as np
import os
from qpu.ast import parse_command
from qpu.qpu_base import QuantumProcessorUnit
from qpu.hilbert import HilbertSpace

ZERO = np.array([1.0, 0.0], dtype=complex)
ONE = np.array([0.0, 1.0], dtype=complex)

CASES = [
    ("0p", "0p", "0p", ZERO, ZERO),
    ("0p", "0p", "1p", ONE, ZERO),
    ("0p", "1p", "0p", ONE, ZERO),
    ("0p", "1p", "1p", ZERO, ONE),
    ("1p", "0p", "0p", ONE, ZERO),
    ("1p", "0p", "1p", ZERO, ONE),
    ("1p", "1p", "0p", ZERO, ONE),
    ("1p", "1p", "1p", ONE, ONE),
]

class TestQuantumFullAdder(unittest.TestCase):
    def setUp(self):
        self.protocol_file = os.path.join(
            os.path.dirname(__file__),
            "SingleBitFullAdder.txt",
        )
        self.proc_name = "SingleBitFullAdder"
        self.basename = "SingleBitFullAdder.txt"

    def run_protocol(self, A: str, B: str, Cin: str):
        class Simulator:
            def __init__(self):
                self.memory = {}
                self.custom_tokens = {}
                self.current_cycle = 0
                self.qpu = QuantumProcessorUnit(num_qubits=5)
                self.hilbert = HilbertSpace()
                self.compiled_processes = {}
                self.subprocess_depth = 0

            def run_cycle(self, suppress_output=False):
                self.current_cycle += 1

        sim = Simulator()

        cwd = os.getcwd()
        proto_dir = os.path.dirname(self.protocol_file)
        os.chdir(proto_dir)

        try:
            compile_cmd = f"COMPILEPROCESS --NAME {self.proc_name} {self.basename}"
            compile_node = parse_command(compile_cmd)
            compile_node.evaluate(
                sim.memory,
                sim.current_cycle,
                sim.qpu,
                sim.hilbert,
                sim,
            )

            call_cmd = f"CALL {self.proc_name} -I {A} {B} {Cin}"
            call_node = parse_command(call_cmd)
            call_node.evaluate(
                sim.memory,
                sim.current_cycle,
                sim.qpu,
                sim.hilbert,
                sim,
            )
        finally:
            os.chdir(cwd)

        sum_state = sim.memory[3][0]
        cout_state = sim.memory[2][0]
        return sum_state, cout_state


def _make_case_test(A: str, B: str, Cin: str, expected_sum, expected_cout):
    def test(self):
        s, c = self.run_protocol(A, B, Cin)
        np.testing.assert_allclose(
            s,
            expected_sum,
            atol=1e-7,
            err_msg=f"SUM mismatch for A={A},B={B},Cin={Cin}",
        )
        np.testing.assert_allclose(
            c,
            expected_cout,
            atol=1e-7,
            err_msg=f"COUT mismatch for A={A},B={B},Cin={Cin}",
        )
    return test


for A, B, Cin, expected_sum, expected_cout in CASES:
    name = f"test_{A}_{B}_{Cin}"
    setattr(
        TestQuantumFullAdder,
        name,
        _make_case_test(A, B, Cin, expected_sum, expected_cout),
    )

if __name__ == "__main__":
    unittest.main()
