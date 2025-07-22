import numpy as np

class SparseState:
    def __init__(self):
        self.amps = {0: 1.0 + 0j}
        self.gen = 0

    def clone(self):
        new = SparseState()
        new.amps = self.amps.copy()
        new.gen = self.gen
        return new

class DenseState:
    def __init__(self, n):
        self.n = n
        self.data = np.zeros((2 ** n,), dtype=np.complex128)
        self.data[0] = 1.0
        self.gen = 0
        self.strides = []
        for k in range(n):
            bit = 1 << k
            low = (
                np.arange(2 ** n, dtype=int)[:: 2 * bit].repeat(bit)
                + np.tile(np.arange(bit), 2 ** (n - k - 1))
            )
            high = low | bit
            self.strides.append((low, high))

    def clone(self):
        new = DenseState(self.n)
        new.data = self.data.copy()
        new.gen = self.gen
        new.strides = self.strides
        return new

class StateVector:
    DENSE_THRESHOLD = 0.05

    def __init__(self, n):
        self.n = n
        self._sparse = SparseState()
        self._dense = None
        self.gen = 0

    @property
    def mode(self):
        return "dense" if self._dense is not None else "sparse"

    def to_dense(self):
        if self._dense is None:
            ds = DenseState(self.n)
            for i, a in self._sparse.amps.items():
                ds.data[i] = a
            self._dense = ds
            self._sparse = None
        return self._dense

    def to_sparse(self):
        if self._dense is not None:
            supp = np.nonzero(np.abs(self._dense.data) > 1e-12)[0]
            if len(supp) <= self.DENSE_THRESHOLD * (2 ** self.n):
                sp = SparseState()
                sp.amps = {i: self._dense.data[i] for i in supp}
                self._dense = None
                self._sparse = sp

    def ensure_cow(self, current_gen):
        if self.gen != current_gen:
            self.clone()
            self.gen = current_gen

    def clone(self):
        if self._dense is not None:
            self._dense = self._dense.clone()
        else:
            self._sparse = self._sparse.clone()

    # -------------------------------------------------------------
    # Gate / measurement helpers
    # -------------------------------------------------------------
    def apply_single_qubit_gate(self, gate, k):
        ds = self.to_dense()
        lo, hi = ds.strides[k]
        a = ds.data[lo]
        b = ds.data[hi]
        ds.data[lo] = gate[0, 0] * a + gate[0, 1] * b
        ds.data[hi] = gate[1, 0] * a + gate[1, 1] * b

    def extract_qubit(self, k):
        ds = self.to_dense()
        lo, hi = ds.strides[k]
        a = ds.data[lo].sum()
        b = ds.data[hi].sum()
        vec = np.array([a, b], dtype=complex)
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        return vec

    def measure_qubit(self, k):
        ds = self.to_dense()
        lo, hi = ds.strides[k]
        a = ds.data[lo]
        b = ds.data[hi]
        p0 = (np.abs(a) ** 2).sum()
        r = np.random.random()
        if r < p0:
            ds.data[hi] = 0
            ds.data[lo] = ds.data[lo] / np.sqrt(p0) if p0 > 0 else ds.data[lo]
            return "0"
        else:
            ds.data[lo] = 0
            ds.data[hi] = ds.data[hi] / np.sqrt(1 - p0) if p0 < 1 else ds.data[hi]
            return "1"

    def full_state(self):
        return self.to_dense().data.copy()
