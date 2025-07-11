import unittest
import numpy as np
import os
from qpu.ast import (
    read_protocol_file,
    parse_parameters,
    substitute_parameters,
    parse_command,
    MainProcessASTNode,
    CycleASTNode,
)
from qpu.qpu_base import QuantumProcessorUnit
from qpu.hilbert import HilbertSpace


class TestQuantumFullAdder(unittest.TestCase):
    def setUp(self):
        self.zero = np.array([1.0, 0.0], dtype=complex)
        self.one  = np.array([0.0, 1.0], dtype=complex)
        # full path to the protocol
        self.protocol_file = os.path.join(
            os.path.dirname(__file__),
            "SingleBitFullAdder.txt"
        )
        self.proc_name = "SingleBitFullAdder"
        self.basename  = "SingleBitFullAdder.txt"

        self.cases = [
            ("0p", "0p", "0p", self.zero, self.zero),
            ("0p", "0p", "1p", self.one, self.zero),  # 0+0+1 → Cout=0
            ("0p", "1p", "0p", self.one, self.zero),
            ("0p", "1p", "1p", self.zero, self.one),
            ("1p", "0p", "0p", self.one, self.zero),
            ("1p", "0p", "1p", self.zero, self.one),
            ("1p", "1p", "0p", self.one, self.zero),  # 1+1+0 → Cout=1
            ("1p", "1p", "1p", self.one, self.one),
        ]

    def run_protocol(self, A: str, B: str, Cin: str):
        # Minimal simulator harness
        class Simulator:
            def __init__(self):
                self.memory               = {}
                self.custom_tokens        = {}
                self.current_cycle        = 0
                self.qpu                  = QuantumProcessorUnit(num_qubits=5)
                self.hilbert              = HilbertSpace()
                self.compiled_processes   = {}
                self.subprocess_depth     = 0

            def run_cycle(self, suppress_output=False):
                self.current_cycle += 1

        sim = Simulator()

        # chdir into the protocol's dir so COMPILEPROCESS can find it by basename
        cwd = os.getcwd()
        proto_dir = os.path.dirname(self.protocol_file)
        os.chdir(proto_dir)

        try:
            # 1) COMPILEPROCESS --NAME SingleBitFullAdder SingleBitFullAdder.txt
            compile_cmd = f"COMPILEPROCESS --NAME {self.proc_name} {self.basename}"
            compile_node = parse_command(compile_cmd)
            compile_node.evaluate(
                sim.memory,
                sim.current_cycle,
                sim.qpu,
                sim.hilbert,
                sim
            )

            # 2) CALL SingleBitFullAdder -I A B Cin
            call_cmd = f"CALL {self.proc_name} -I {A} {B} {Cin}"
            call_node = parse_command(call_cmd)
            # we can ignore the returned transcript; all state is in sim.memory
            call_node.evaluate(
                sim.memory,
                sim.current_cycle,
                sim.qpu,
                sim.hilbert,
                sim
            )
        finally:
            os.chdir(cwd)

        # The protocol measures SUM into physical qubit 3 at cycle 0
        # and COUT into qubit 2 at cycle 0
        sum_state  = sim.memory[3][0]
        cout_state = sim.memory[2][0]
        return sum_state, cout_state

    def test_all_combinations(self):
        for A, B, Cin, expected_sum, expected_cout in self.cases:
            with self.subTest(A=A, B=B, Cin=Cin):
                s, c = self.run_protocol(A, B, Cin)
                np.testing.assert_allclose(
                    s, expected_sum, atol=1e-7,
                    err_msg=f"SUM mismatch for A={A},B={B},Cin={Cin}"
                )
                np.testing.assert_allclose(
                    c, expected_cout, atol=1e-7,
                    err_msg=f"COUT mismatch for A={A},B={B},Cin={Cin}"
                )


if __name__ == "__main__":
    unittest.main()
