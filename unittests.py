import unittest
import numpy as np
import os
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
        parse_command(compile_cmd).evaluate(sim.memory, sim.current_cycle, sim.qpu, sim.hilbert, sim)
        call_args = ' '.join(params)
        call_cmd = f"CALL {proc_name} -I {call_args}"
        parse_command(call_cmd).evaluate(sim.memory, sim.current_cycle, sim.qpu, sim.hilbert, sim)
    finally:
        os.chdir(cwd)
    return sim


class TestSingleBitAdder(unittest.TestCase):
    def setUp(self):
        self.zero = np.array([1.0, 0.0], dtype=complex)
        self.one = np.array([0.0, 1.0], dtype=complex)
        self.proc_file = os.path.join(os.path.dirname(__file__), "SingleBitFullAdder.txt")
        self.proc_name = "SingleBitFullAdder"
        self.cases = [
            ("0p", "0p", "0p", self.zero, self.zero),
            ("0p", "0p", "1p", self.one, self.zero),
            ("0p", "1p", "0p", self.one, self.zero),
            ("0p", "1p", "1p", self.zero, self.one),
            ("1p", "0p", "0p", self.one, self.zero),
            ("1p", "0p", "1p", self.zero, self.one),
            ("1p", "1p", "0p", self.one, self.zero),
            ("1p", "1p", "1p", self.one, self.one),
        ]

    def test_all_combinations(self):
        for A, B, Cin, exp_sum, exp_cout in self.cases:
            with self.subTest(A=A, B=B, Cin=Cin):
                sim = run_process(self.proc_file, self.proc_name, [A, B, Cin])
                s = sim.memory[3][0]
                c = sim.memory[2][0]
                np.testing.assert_allclose(s, exp_sum, atol=1e-7)
                np.testing.assert_allclose(c, exp_cout, atol=1e-7)


class TestTwoBitAdder(unittest.TestCase):
    def setUp(self):
        self.zero = np.array([1.0, 0.0], dtype=complex)
        self.one = np.array([0.0, 1.0], dtype=complex)
        self.proc_file = os.path.join(os.path.dirname(__file__), "TwoBitFullAdder.txt")
        self.proc_name = "TwoBitFullAdder"

    def expected(self, A0, A1, B0, B1, Cin):
        a = (int(A1[-2]) << 1) | int(A0[-2])
        b = (int(B1[-2]) << 1) | int(B0[-2])
        cin = int(Cin[-2])
        total = a + b + cin
        s0 = (total & 1)
        s1 = (total >> 1) & 1
        cout = (total >> 2) & 1
        return (self.one if s0 else self.zero,
                self.one if s1 else self.zero,
                self.one if cout else self.zero)

    def test_all_combinations(self):
        bits = ["0p", "1p"]
        for A0 in bits:
            for A1 in bits:
                for B0 in bits:
                    for B1 in bits:
                        for Cin in bits:
                            with self.subTest(A0=A0, A1=A1, B0=B0, B1=B1, Cin=Cin):
                                sim = run_process(self.proc_file, self.proc_name,
                                                  [A0, A1, B0, B1, Cin])
                                s0 = sim.memory["Sum0"][0]
                                s1 = sim.memory["Sum1"][1]
                                cout = sim.memory["Cout"][1]
                                exp_s0, exp_s1, exp_cout = self.expected(A0, A1, B0, B1, Cin)
                                np.testing.assert_allclose(s0, exp_s0, atol=1e-7)
                                np.testing.assert_allclose(s1, exp_s1, atol=1e-7)
                                np.testing.assert_allclose(cout, exp_cout, atol=1e-7)


if __name__ == "__main__":
    unittest.main()

