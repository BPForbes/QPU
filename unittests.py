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
        parse_command(f"COMPILEPROCESS --NAME {proc_name} {os.path.basename(proc_file)}") \
            .evaluate(sim.memory, sim.current_cycle, sim.qpu, sim.hilbert, sim)
        parse_command(f"CALL {proc_name} -I {' '.join(params)}") \
            .evaluate(sim.memory, sim.current_cycle, sim.qpu, sim.hilbert, sim)
    finally:
        os.chdir(cwd)
    return sim

class TestTwoBitAdder(unittest.TestCase):
    def setUp(self):
        self.zero = np.array([1.0, 0.0], dtype=complex)
        self.one  = np.array([0.0, 1.0], dtype=complex)
        self.proc_file = os.path.join(os.path.dirname(__file__), "TwoBitFullAdder.txt")
        self.proc_name = "TwoBitFullAdder"
        self.N = 2

    def expected(self, a, b, cin):
        total = a + b + cin
        # sum bits s0, s1 and carry out
        s0   = (total >> 0) & 1
        s1   = (total >> 1) & 1
        cout = (total >> 2) & 1
        return (self.one if s0 else self.zero,
                self.one if s1 else self.zero,
                self.one if cout else self.zero)

    def test_all_combinations(self):
        for a in range(2**self.N):
            for b in range(2**self.N):
                for cin in (0, 1):
                    with self.subTest(a=a, b=b, cin=cin):
                        # format inputs as "0p"/"1p"
                        params = [f"{(a>>i)&1}p" for i in range(self.N)]
                        params += [f"{(b>>i)&1}p" for i in range(self.N)]
                        params.append(f"{cin}p")

                        sim = run_process(self.proc_file, self.proc_name, params)
                        s0_cycle = max(sim.memory["Sum0"].keys())
                        s1_cycle = max(sim.memory["Sum1"].keys())
                        cout_cycle = max(sim.memory["Cout"].keys())

                        s0, s1, cout = (
                            sim.memory["Sum0"][s0_cycle],
                            sim.memory["Sum1"][s1_cycle],
                            sim.memory["Cout"][cout_cycle]
                        )
                        exp_s0, exp_s1, exp_cout = self.expected(a, b, cin)

                        np.testing.assert_allclose(s0, exp_s0, atol=1e-7)
                        np.testing.assert_allclose(s1, exp_s1, atol=1e-7)
                        np.testing.assert_allclose(cout, exp_cout, atol=1e-7)
