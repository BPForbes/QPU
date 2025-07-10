# qpu_base.py

import numpy as np
import math
import secrets

# Standard 2×2 Pauli / Hadamard matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def format_qubit_state(state: np.ndarray, tol=1e-6) -> str:
    """
    Nicely format a 1- or multi-qubit state vector for debugging.
    """
    if state.ndim != 1:
        return str(state)
    d = state.shape[0]
    if d == 2:
        def fmt(z):
            re, im = z.real, z.imag
            if abs(im) < tol:
                return f"{re:.3f}"
            if abs(re) < tol:
                return f"{im:.3f}j"
            sign = '+' if im >= 0 else '-'
            return f"{re:.3f}{sign}{abs(im):.3f}j"
        return f"{fmt(state[0])}|0⟩ + {fmt(state[1])}|1⟩"
    else:
        n = int(math.log2(d))
        terms = []
        for k, amp in enumerate(state):
            if abs(amp) < tol: continue
            bits = format(k, f"0{n}b")
            terms.append(f"{amp:.3f}|{bits}⟩")
        return " + ".join(terms)


def embed_registers(*regs):
    """
    Tensor-up smaller registers with |0⟩ to match the largest one.
    """
    dims = [r.shape[0] for r in regs]
    max_dim = max(dims)
    new_regs = []
    for r in regs:
        if r.shape[0] == max_dim:
            new_regs.append(r.copy())
        else:
            extra = int(math.log2(max_dim)) - int(math.log2(r.shape[0]))
            zero = np.array([1, 0], dtype=complex)
            up = r.copy()
            for _ in range(extra):
                up = np.kron(up, zero)
            new_regs.append(up)
    return tuple(new_regs), max_dim


class QuantumProcessorUnit:
    """
    Simulates a QPU with N physical qubits and three special custom qubits:
      - "0p" fixed at |0⟩
      - "1p" fixed at |1⟩
      - "sp" a random superposition
    Provides single-, double- and triple-controlled gates, plus common Boolean
    primitives mapped to CCNOT and CNOT.
    """

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits

        # Global state vector |0...0⟩
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1.0

        # Individual physical qubit registers
        self.local_states = [np.array([1, 0], dtype=complex)
                             for _ in range(num_qubits)]

        # Custom (string-keyed) registers
        self.custom_states = {}

        # Preload the three special qubits:
        self.custom_states["0p"] = np.array([1, 0], dtype=complex)
        self.custom_states["1p"] = np.array([0, 1], dtype=complex)
        # random superposition
        rng = secrets.SystemRandom()
        a = rng.random() + 1j*rng.random()
        b = rng.random() + 1j*rng.random()
        vec = np.array([a, b], dtype=complex)
        self.custom_states["sp"] = vec / np.linalg.norm(vec)

        # Device properties fingerprint
        # Column 0: frequency (GHz), cols 1,2: T1,T2 (μs)
        self.property_table = np.zeros((num_qubits, 3))
        self.property_table[:, 0] = np.random.uniform(4.9, 5.1, num_qubits)
        self.property_table[:, 1] = np.abs(np.random.normal(100, 10, num_qubits))
        self.property_table[:, 2] = np.abs(np.random.normal( 80, 10, num_qubits))

    def _validate_qubit(self, *idxs):
        """
        Each idx is either 0 ≤ int < num_qubits or a string in custom_states.
        """
        for i in idxs:
            if isinstance(i, int):
                if not (0 <= i < self.num_qubits):
                    raise IndexError(f"Qubit index {i} out of range")
            elif isinstance(i, str):
                if i not in self.custom_states:
                    raise ValueError(f"Unknown custom qubit '{i}'")
            else:
                raise TypeError(f"Invalid qubit index: {i!r}")

    def _get_register(self, key, target_dim=None):
        """
        Fetch the (possibly multi-dim) state vector for `key`.
        Optionally split off trailing |0...0⟩ if reg_dim > target_dim.
        """
        if isinstance(key, int):
            reg = self.local_states[key]
        else:
            reg = self.custom_states[key]

        if target_dim is not None and reg.shape[0] > target_dim:
            reg = self.split_register(key, reg, target_dim)
            self._update_register(key, reg)

        return reg

    def _update_register(self, key, vec):
        """
        Write back a new state vector for `key`.
        """
        if isinstance(key, int):
            self.local_states[key] = vec
        else:
            self.custom_states[key] = vec

    def split_register(self, key, state, target_dim):
        """
        If state = (target_dim)-dim ⊗ |0...0⟩, drop the trailing |0...0⟩ factor.
        """
        cd = state.shape[0]
        if cd == target_dim:
            return state
        if target_dim >= cd or (cd & (cd-1)) or (target_dim & (target_dim-1)):
            raise ValueError("Invalid split dimensions")

        ncur = int(math.log2(cd))
        ntar = int(math.log2(target_dim))
        extra = ncur - ntar
        mat = state.reshape(target_dim, 2**extra)
        # ensure columns beyond 0 are zero
        if not np.allclose(mat[:,1:], 0, atol=1e-8):
            raise ValueError("Cannot split: entangled or non-|0⟩ tail")
        col0 = mat[:,0]
        return col0 / np.linalg.norm(col0)

    def apply_single_qubit_gate(self, gate, target, sub_index=0):
        """
        Apply a 2×2 gate to `target`.  Updates the global state if `target` is int.
        """
        self._validate_qubit(target)
        reg = self._get_register(target, target_dim=2)

        # Update global if physical qubit
        if isinstance(target, int):
            mask = 1 << target
            N = self.state.size
            for i in range(N):
                if (i & mask) == 0:
                    j = i | mask
                    a, b = self.state[i], self.state[j]
                    self.state[i] = gate[0,0]*a + gate[0,1]*b
                    self.state[j] = gate[1,0]*a + gate[1,1]*b

        new_reg = gate @ reg
        self._update_register(target, new_reg)
        return new_reg

    def apply_controlled_gate(self, gate, control, target):
        """
        Single-controlled gate.  Fast-path for int,int; otherwise on-register.
        """
        self._validate_qubit(control, target)
        cs = self._get_register(control)
        ts = self._get_register(target)
        (cs, ts), _ = embed_registers(cs, ts)
        self._update_register(control, cs)
        self._update_register(target, ts)

        if isinstance(control, int) and isinstance(target, int):
            c_mask = 1 << control
            t_mask = 1 << target
            N = self.state.size
            for i in range(N):
                if (i & c_mask) == 0:
                    continue
                j = i ^ t_mask
                a, b = self.state[i], self.state[j]
                self.state[i] = gate[0,0]*a + gate[0,1]*b
                self.state[j] = gate[1,0]*a + gate[1,1]*b
            # also update the single-qubit register
            new_t = gate @ self.local_states[target]
            self._update_register(target, new_t)
            return new_t
        else:
            return self.apply_single_qubit_gate(gate, target)

    def apply_cnot(self, control, target):
        return self.apply_controlled_gate(X, control, target)

    def apply_ccnot(self, control1, control2, target):
        """
        Toffoli: flips `target` iff both controls are |1⟩.
        """
        self._validate_qubit(control1, control2, target)
        s1 = self._get_register(control1)
        s2 = self._get_register(control2)
        st = self._get_register(target)
        (s1, s2, st), _ = embed_registers(s1, s2, st)
        self._update_register(control1, s1)
        self._update_register(control2, s2)
        self._update_register(target, st)

        if all(isinstance(x, int) for x in (control1, control2, target)):
            m1 = 1 << control1
            m2 = 1 << control2
            mt = 1 << target
            N = self.state.size
            for i in range(N):
                if (i & m1) and (i & m2):
                    j = i ^ mt
                    self.state[i], self.state[j] = self.state[j], self.state[i]
            # refresh the single-qubit register
            return self._get_register(target)
        else:
            # fallback: X on target
            return self.apply_single_qubit_gate(X, target)

    # ----------------------------------------------------------------
    # Built-in Boolean primitives (no explicit target argument):
    # ----------------------------------------------------------------

    def apply_not(self, a):
        """
        Compute NOT(A) onto the special |1p⟩ register:
          CCNOT(A, "1p", "1p") flips "1p" ─→ |1⟩⊕A = NOT(A).
        """
        # reset "1p" to |1>
        self.custom_states["1p"] = np.array([0,1], dtype=complex)
        return self.apply_ccnot(a, "1p", "1p")

    def apply_and(self, a, b):
        """
        Compute A∧B onto the special |0p⟩ register:
          CCNOT(A, B, "0p") flips "0p" from |0> to |1> exactly when A=B=1.
        """
        self.custom_states["0p"] = np.array([1,0], dtype=complex)
        return self.apply_ccnot(a, b, "0p")

    def apply_nand(self, a, b):
        """
        Compute NAND(A,B) onto "1p":
          CCNOT(A, B, "1p") flips "1p" from |1>→|0> iff A=B=1.
        """
        self.custom_states["1p"] = np.array([0,1], dtype=complex)
        return self.apply_ccnot(a, b, "1p")

    def apply_xor(self, a, b):
        """
        Compute A⊕B onto "0p" via two CNOTs:
          CNOT(A, "0p");  CNOT(B, "0p")
        """
        self.custom_states["0p"] = np.array([1,0], dtype=complex)
        self.apply_cnot(a, "0p")
        self.apply_cnot(b, "0p")
        return self._get_register("0p")

    def apply_or(self, a, b):
        """
        Compute A∨B onto "0p":
          CNOT(A, "0p"); CNOT(B, "0p"); CCNOT(A, B, "0p")
        """
        self.custom_states["0p"] = np.array([1,0], dtype=complex)
        self.apply_cnot(a, "0p")
        self.apply_cnot(b, "0p")
        self.apply_ccnot(a, b, "0p")
        return self._get_register("0p")

    def apply_phase(self, angle, target):
        P = np.array([[1, 0],
                      [0, np.exp(1j*angle)]], dtype=complex)
        return self.apply_single_qubit_gate(P, target)

    # ----------------------------------------------------------------
    # Measurement
    # ----------------------------------------------------------------

    def measure_qubit(self, target):
        """
        Collapse and measure a single physical or custom qubit.
        Returns ('0'|'1', new_state).
        """
        reg = self._get_register(target, target_dim=2)
        p0 = abs(reg[0])**2
        if secrets.SystemRandom().random() < p0:
            new = np.array([1,0], dtype=complex)
            self._update_register(target, new)
            return "0", new
        else:
            new = np.array([0,1], dtype=complex)
            self._update_register(target, new)
            return "1", new

    def measure_register(self, key):
        """
        Collapse a multi-qubit register (power-of-two dimension) to one basis state.
        Returns (bitstring, new_state).
        """
        state = self._get_register(key)
        dim = state.shape[0]
        n = int(math.log2(dim))
        probs = np.abs(state)**2
        probs /= probs.sum()
        idx = int(np.random.choice(dim, p=probs))
        bits = format(idx, f"0{n}b")
        collapsed = np.zeros(dim, dtype=complex)
        collapsed[idx] = 1.0
        self._update_register(key, collapsed)
        return bits, collapsed

    # ----------------------------------------------------------------
    # Device‐fingerprint methods
    # ----------------------------------------------------------------

    def get_frequency_fingerprint(self) -> np.ndarray:
        """
        Return the frequency vector (GHz) for all physical qubits.
        """
        return self.property_table[:, 0].copy()

    def get_t1t2_fingerprint(self) -> np.ndarray:
        """
        Return the flattened [T1,T2,...] vector (μs).
        """
        return self.property_table[:, 1:3].flatten().copy()


def compare_frequency_fingerprints(f1: np.ndarray,
                                   f2: np.ndarray,
                                   delta_avg: float) -> float:
    """
    Normalized Hamming distance: fraction of qubits whose |f1−f2|>delta_avg.
    """
    diff = np.abs(f1 - f2)
    num = int((diff > delta_avg).sum())
    return num / len(f1)


def apply_gate_on_register(state: np.ndarray,
                           gate: np.ndarray,
                           sub_index: int = 0) -> np.ndarray:
    """
    Apply a 2×2 gate to one sub-qubit (at position sub_index) within a larger register.
    """
    n = int(math.log2(state.shape[0]))
    op = 1
    for i in range(n):
        op = np.kron(op, gate if i == sub_index else I)
    return op @ state
