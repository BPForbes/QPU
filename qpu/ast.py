# qpu/ast.py

"""
Implements an abstract syntax tree (AST) for representing quantum operations
using a custom -I syntax with colon-based cycle tokens, plus support for
MAIN-PROCESS / DECLARECHILD / RUNCHILD / RETURNVALS / ACCEPTVALS and
Unix-style "\" line continuations.
"""

import os
import hashlib
import re
import random
import secrets

import numpy as np
from qpu.qpu_base import QuantumProcessorUnit, X, Y, Z, H, format_qubit_state

# ------------------------------------------------------------------------
# Helper: complete memory snapshot (for subprocesses)
# ------------------------------------------------------------------------
def generate_complete_memory_snapshot(simulator) -> str:
    snapshot_lines = []
    keys = set(simulator.memory.keys())
    if hasattr(simulator.qpu, "custom_states"):
        keys |= set(simulator.qpu.custom_states.keys())

    def sort_key(k):
        try:
            return (0, int(k))
        except:
            return (1, str(k))

    for k in sorted(keys, key=sort_key):
        entries = []
        if k in simulator.memory:
            for c in sorted(simulator.memory[k].keys()):
                raw = simulator.memory[k][c]
                s = raw[1] if isinstance(raw, tuple) else raw
                if isinstance(s, np.ndarray):
                    body = format_qubit_state(s)
                elif isinstance(s, list):
                    bullets = []
                    for elt in s:
                        if isinstance(elt, np.ndarray):
                            bullets.append(format_qubit_state(elt))
                        elif isinstance(elt, dict):
                            for name, val in elt.items():
                                bullets.append(f"{name}: {val}")
                        else:
                            bullets.append(str(elt))
                    body = "\n    - " + "\n    - ".join(bullets)
                elif isinstance(s, dict):
                    bullets = [f"{name}: {state}" for name, state in s.items()]
                    body = "\n    - " + "\n    - ".join(bullets)
                else:
                    body = str(s)
                entries.append((c, body))
        if hasattr(simulator.qpu, "custom_states") and k in simulator.qpu.custom_states:
            s = simulator.qpu.custom_states[k]
            body = format_qubit_state(s)
            entries.append(("(custom)", body))
        for c, body in entries:
            if "\n    -" in body:
                snapshot_lines.append(f"{k}@{c}:")
                snapshot_lines.append(body)
            else:
                snapshot_lines.append(f"{k}@{c}: {body}")
    return "\n".join(snapshot_lines)


# ------------------------------------------------------------------------
# Utilities: file parsing, parameter substitution, fingerprinting, etc.
# ------------------------------------------------------------------------
def read_protocol_file(filename: str):
    joined = []
    buffer = ""
    with open(filename, "r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.endswith("\\"):
                buffer += line[:-1].rstrip() + " "
            else:
                buffer += line
                joined.append(buffer)
                buffer = ""
        if buffer:
            joined.append(buffer)

    processed, in_comment = [], False
    for raw in joined:
        line = raw.strip()
        if not line:
            continue
        if "/*" in line:
            i = line.find("/*")
            prefix = line[:i].strip()
            j = line.find("*/", i + 2)
            if j != -1:
                suffix = line[j + 2 :].strip()
                line = (prefix + " " + suffix).strip()
            else:
                line = prefix
                in_comment = True
            if not line:
                continue
        if in_comment:
            if "*/" in line:
                line = line.split("*/", 1)[1].strip()
                in_comment = False
            else:
                continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if line:
            processed.append(line)
    return processed


def parse_parameters(def_line: str):
    if not def_line.upper().startswith("PARAMS:"):
        return []
    parts = def_line.split(":", 1)[1].strip().split()
    params = []
    for part in parts:
        if ":" not in part:
            continue
        name, tstr = part.split(":", 1)
        params.append((name.strip(), tstr.strip()))
    return params


def assign_parameter_values(param_defs, provided_values):
    assignments = {}
    for i, (name, _) in enumerate(param_defs):
        if i < len(provided_values):
            assignments[name] = provided_values[i]
    return assignments


def substitute_parameters(lines, assignments):
    pattern = re.compile(r"\$([A-Za-z0-9_]+)")
    out = []
    for line in lines:
        out.append(
            pattern.sub(lambda m: str(assignments.get(m.group(1), m.group(0))), line)
        )
    return out


def binary_fingerprint(content: bytes) -> str:
    return "".join(format(b, "08b") for b in content)


def binary_to_ascii(binary_str: str) -> str:
    table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    rem = len(binary_str) % 6
    if rem:
        binary_str += "0" * (6 - rem)
    return "".join(
        table[int(binary_str[i : i + 6], 2)] for i in range(0, len(binary_str), 6)
    )


def generate_noise(params_str: str, noise_length=64) -> str:
    seed = int(hashlib.sha256(params_str.encode("utf-8")).hexdigest(), 16)
    rnd = random.Random(seed)
    return "".join(rnd.choice("01") for _ in range(noise_length))


def interpret_token(token: str):
    """
    Recognizes:
      - numeric → physical qubit index
      - '0p','1p','sp' → built-in custom registers
      - otherwise → custom token name
    """
    t = re.sub(r"(?i)_dim\d+", "", token).lstrip("\ufeff")
    u = t.upper()
    if u in ("0P", "1P", "SP"):
        return t.lower()
    try:
        return int(token)
    except ValueError:
        return token


def resolve_state(val):
    if isinstance(val, tuple) and len(val) == 2 and hasattr(val[1], "shape"):
        return val[1]
    return val


def readdress_value(memory, key, requested_cycle, qpu=None):
    if key in memory:
        cycles = sorted(memory[key].keys())
        candidate = None
        for c in cycles:
            if requested_cycle is None or c <= requested_cycle:
                candidate = c
            else:
                break
        if candidate is None:
            raise ValueError(f"No value for key {key}@≤{requested_cycle}")
        v = memory[key][candidate]
        return v[1] if isinstance(v, tuple) else v
    if qpu and hasattr(qpu, "custom_states") and key in qpu.custom_states:
        return qpu.custom_states[key]
    raise KeyError(f"Key {key} not found")


# ------------------------------------------------------------------------
# Gate Handlers (for the *primitive* gates in qpu_base)
# ------------------------------------------------------------------------
gate_handlers = {
    "X": {
        "inputs": 1,
        "handler": lambda q, ins, out, param: (
            f"X on {ins[0]}",
            q.apply_single_qubit_gate(X, ins[0]),
        ),
    },
    "H": {
        "inputs": 1,
        "handler": lambda q, ins, out, param: (
            f"H on {ins[0]}",
            q.apply_single_qubit_gate(H, ins[0]),
        ),
    },
    "CNOT": {
        "inputs": 1,
        "handler": lambda q, ins, out, param: (
            f"CNOT {ins[0]}→{out}",
            q.apply_controlled_gate(X, ins[0], out),
        ),
    },
    "CCNOT": {
        "inputs": 2,
        "handler": lambda q, ins, out, param: (
            f"CCNOT {ins[0]},{ins[1]}→{out}",
            q.apply_ccnot(ins[0], ins[1], out),
        ),
    },
    "PHASE": {
        "inputs": 1,
        "handler": lambda q, ins, out, param: (
            f"PHASE({param}) on {ins[0]}",
            q.apply_phase(param, ins[0]),
        ),
    },
}


# ------------------------------------------------------------------------
# AST Node Classes
# ------------------------------------------------------------------------

class CycleASTNode:
    def __init__(self, tokens):
        if len(tokens) != 2:
            raise ValueError("CYCLE requires exactly 1 argument")
        self.n = int(tokens[1])

    def is_ready(self, *args):
        return True

    def evaluate(self, memory, current_cycle, qpu, hilbert, simulator=None):
        if simulator is None or not hasattr(simulator, "run_cycle"):
            raise ValueError("CYCLE requires a simulator")
        for _ in range(self.n):
            simulator.run_cycle(suppress_output=True)
        return f"Advanced {self.n} cycles", None

    def __repr__(self):
        return f"CYCLE {self.n}"


class MainProcessASTNode:
    """AST node for MAIN-PROCESS <name>."""
    def __init__(self, tokens):
        if len(tokens) != 2:
            raise ValueError("MAIN-PROCESS requires exactly one name")
        self.name = tokens[1]

    def is_ready(self, *args):
        return True

    def evaluate(self, memory, current_cycle, qpu, hilbert, simulator=None):
        simulator.current_main = self.name
        return f"Main process '{self.name}' started", None

    def __repr__(self):
        return f"MAIN-PROCESS {self.name}"

class CreateTokenASTNode:
    def __init__(self, tokens):
        # Syntax: CREATETOKEN -I name1 name2 ...
        if not tokens or tokens[0].upper() != "-I":
            raise ValueError("CREATETOKEN requires -I followed by token names")
        self.names = tokens[1:]
        if not self.names:
            raise ValueError("CREATETOKEN requires at least one token name after -I")

    def is_ready(self, *args):
        return True

    def evaluate(self, memory, current_cycle, qpu, hilbert, simulator=None):
        if simulator is None:
            raise ValueError("CREATETOKEN requires a simulator")
        # initialize custom_tokens dict on simulator if missing
        if not hasattr(simulator, "custom_tokens"):
            simulator.custom_tokens = {}

        for name in self.names:
            # create the register at |0> state if needed
            if name not in qpu.custom_states:
                qpu.custom_states[name] = np.array([1, 0], dtype=complex)
            # record in memory and Hilbert outputs
            memory.setdefault(name, {})[0] = qpu.custom_states[name]
            hilbert.output(name, 0, qpu.custom_states[name])
            # track in simulator
            simulator.custom_tokens[name] = name

        return f"CREATETOKEN created: {', '.join(self.names)}", None

    def __repr__(self):
        return "CREATETOKEN -I " + " ".join(self.names)


class DeleteTokenASTNode:
    def __init__(self, tokens):
        # Syntax: DELETETOKEN -I name1 name2 ...
        if not tokens or tokens[0].upper() != "-I":
            raise ValueError("DELETETOKEN requires -I followed by token names")
        self.names = tokens[1:]
        if not self.names:
            raise ValueError("DELETETOKEN requires at least one token name after -I")

    def is_ready(self, *args):
        return True

    def evaluate(self, memory, current_cycle, qpu, hilbert, simulator=None):
        if simulator is None:
            raise ValueError("DELETETOKEN requires a simulator")
        # initialize custom_tokens dict if missing
        if not hasattr(simulator, "custom_tokens"):
            simulator.custom_tokens = {}

        for name in self.names:
            if name not in simulator.custom_tokens:
                raise ValueError(f"Token '{name}' not found")
            # remove from qpu and memory
            qpu.custom_states.pop(name, None)
            memory.pop(name, None)
            # remove from simulator
            simulator.custom_tokens.pop(name, None)

        return f"DELETETOKEN deleted: {', '.join(self.names)}", None

    def __repr__(self):
        return "DELETETOKEN -I " + " ".join(self.names)


class SetASTNode:
    def __init__(self, tokens):
        if len(tokens) != 2:
            raise ValueError("SET requires 2 tokens")
        addr, val = tokens
        parts = addr.split(":")
        if len(parts) != 2:
            raise ValueError("SET address must be <q>:<cycle>")
        k = int(parts[0]) if parts[0].isdigit() else parts[0]
        self.key, self.cy, self.val = k, int(parts[1]), val

    def is_ready(self, *args):
        return True

    def evaluate(self, memory, current_cycle, qpu, hilbert, simulator=None):
        if simulator is not None and not hasattr(simulator, "custom_tokens"):
            simulator.custom_tokens = {}

        m = re.match(r"^(0p|1p|sp)(?:_dim(\d+))?$", self.val.lower())
        if not m:
            raise ValueError(f"Invalid SET value: {self.val}")
        base, dim_s = m.groups()
        base = base.lower()
        if base == "0p":
            state = np.array([1.0, 0.0], dtype=complex)
        elif base == "1p":
            state = np.array([0.0, 1.0], dtype=complex)
        else:
            amp0 = secrets.SystemRandom().random() + 1j * secrets.SystemRandom().random()
            amp1 = secrets.SystemRandom().random() + 1j * secrets.SystemRandom().random()
            vec = np.array([amp0, amp1], dtype=complex)
            state = vec / np.linalg.norm(vec)

        if dim_s:
            K = int(dim_s)
            zero = np.array([1.0, 0.0], dtype=complex)
            for _ in range(K - 1):
                state = np.kron(state, zero)

        memory.setdefault(self.key, {})[self.cy] = state
        hilbert.output(self.key, self.cy, state)

        if isinstance(self.key, int) and self.key < qpu.num_qubits:
            qpu.local_states[self.key] = state
        else:
            qpu.custom_states[self.key] = state
            simulator.custom_tokens[self.key] = self.key

        return f"SET {self.key}@{self.cy} → {format_qubit_state(state)}", state

    def __repr__(self):
        return f"SET {self.key}:{self.cy} {self.val}"


class CompileASTNode:
    def __init__(self, tokens):
        U = [t.upper() for t in tokens]
        self.no_params = "-$R" in U
        filtered = [t for t in tokens if t.upper() not in ("-I", "-$R")]
        if len(filtered) < 4:
            raise ValueError("COMPILEPROCESS requires --NAME, name, filename")
        self.proc = filtered[2]
        self.file = filtered[3]
        self.params = filtered[4:]

    def is_ready(self, *args):
        return True

    def evaluate(self, memory, current_cycle, qpu, hilbert, simulator=None):
        if simulator is None:
            raise ValueError("COMPILEPROCESS requires a simulator")

        path = os.path.join(os.getcwd(), self.file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File '{self.file}' not found")

        lines = read_protocol_file(path)
        defs = []
        if lines and lines[0].upper().startswith("PARAMS:"):
            defs = parse_parameters(lines[0])
            lines = lines[1:]

        if not self.no_params and self.params:
            assigns = assign_parameter_values(defs, self.params)
            lines = substitute_parameters(lines, assigns)

        for L in lines:
            up = L.strip().upper()
            if up.startswith(("CALL ", "RUNCHILD ", "DECLARECHILD ")):
                dep = L.strip().split()[1]
                if dep not in simulator.compiled_processes:
                    CompileASTNode(
                        ["COMPILEPROCESS", "--NAME", dep, f"{dep}.txt"]
                    ).evaluate(memory, current_cycle, qpu, hilbert, simulator)

        noise = generate_noise(" ".join(self.params))
        content = ("\n".join(self.params + lines + [noise])).encode("utf-8")
        fp = binary_to_ascii(binary_fingerprint(content))

        errs = []
        for idx, L in enumerate(lines, 1):
            t = L.strip()
            if not t or t.startswith(("#", "/*")) or t.upper().startswith("PARAMS:"):
                continue
            try:
                parse_command(L)
            except Exception as e:
                errs.append(f"Line {idx}: {L} → {e}")

        if errs:
            raise ValueError(
                f"Compilation failed for '{self.proc}':\n" + "\n".join(errs)
            )

        simulator.compiled_processes[self.proc] = {
            "lines": lines,
            "fingerprint": fp,
            "param_defs": defs,
            "no_params": self.no_params,
        }

        return (
            f"COMPILEPROCESS '{self.proc}' compiled ({len(lines)} lines) fingerprint={fp}",
            None,
        )

    def __repr__(self):
        s = f"COMPILEPROCESS --NAME {self.proc} {self.file}"
        if self.params:
            s += " -I " + " ".join(self.params)
        if self.no_params:
            s += " -$R"
        return s


class CreateTokenASTNode:
    def __init__(self, tokens):
        if tokens and tokens[0].upper() == "-I":
            tokens = tokens[1:]
        if not tokens:
            raise ValueError("CREATETOKEN requires at least one token")
        self.names = tokens[:]

    def is_ready(self, *args):
        return True

    def evaluate(self, memory, current_cycle, qpu, hilbert, simulator=None):
        for name in self.names:
            if name not in qpu.custom_states:
                qpu.custom_states[name] = np.array([1, 0], dtype=complex)
            if simulator is not None:
                simulator.custom_tokens[name] = name
            memory.setdefault(name, {})[0] = qpu.custom_states[name]
            hilbert.output(name, 0, qpu.custom_states[name])

        return f"CREATETOKEN created: {', '.join(self.names)}", None

    def __repr__(self):
        return "CREATETOKEN -I " + " ".join(self.names)


class DeleteTokenASTNode:
    def __init__(self, tokens):
        if tokens and tokens[0].upper() == "-I":
            tokens = tokens[1:]
        if not tokens:
            raise ValueError("DELETETOKEN requires at least one token")
        self.names = tokens[:]

    def is_ready(self, *args):
        return True

    def evaluate(self, memory, current_cycle, qpu, hilbert, simulator=None):
        for name in self.names:
            if simulator and name not in simulator.custom_tokens:
                raise ValueError(f"Token '{name}' not found")
            if name in qpu.custom_states:
                del qpu.custom_states[name]
            if simulator:
                del simulator.custom_tokens[name]
        return f"DELETETOKEN deleted: {', '.join(self.names)}", None

    def __repr__(self):
        return "DELETETOKEN -I " + " ".join(self.names)


class FreeASTNode:
    def __init__(self, tokens):
        self.inputs = []
        for tok in tokens:
            parts = tok.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid FREE token '{tok}'")
            k = int(parts[0]) if parts[0].isdigit() else parts[0]
            self.inputs.append((k, int(parts[1])))

    def is_ready(self, *args):
        return True

    def evaluate(self, memory, current_cycle, qpu, hilbert, simulator=None):
        if simulator is None or not hasattr(simulator, "free_memory_entries"):
            raise ValueError("FREE requires simulator.free_memory_entries()")
        msg = simulator.free_memory_entries(self.inputs)
        return msg, None

    def __repr__(self):
        return "FREE -I " + " ".join(f"{k}:{c}" for k, c in self.inputs)


class MeasureASTNode:
    """
    MEASURE:
      - without -I: collapse every register now
      - with    -I: collapse the specified register q or q:cycle
    """
    def __init__(self, input_token: str = None):
        self.input_token = input_token

    def is_ready(self, memory, current_cycle, qpu=None):
        return True

    def evaluate(self, memory, current_cycle, qpu, hilbert, simulator=None):
        def _commit(key, cycle, vec):
            memory.setdefault(key, {})[cycle] = vec
            hilbert.output(key, cycle, vec)

        if self.input_token:
            parts = self.input_token.split(":")
            if len(parts) == 2:
                k_str, cy_str = parts
                k = int(k_str) if k_str.isdigit() else k_str
                cy = int(cy_str)
            elif len(parts) == 1:
                k = int(parts[0]) if parts[0].isdigit() else parts[0]
                cy = current_cycle
            else:
                raise ValueError(f"Invalid MEASURE token: '{self.input_token}'")

            bits, vec = qpu.measure_register(k)
            _commit(k, cy, vec)
            return f"MEASURE {k}@{cy} → {bits}", vec

        results = {}
        for q in range(qpu.num_qubits):
            bits, vec = qpu.measure_register(q)
            _commit(q, current_cycle, vec)
            results[f"q{q}"] = bits
        for tok in list(qpu.custom_states):
            bits, vec = qpu.measure_register(tok)
            _commit(tok, current_cycle, vec)
            results[str(tok)] = bits

        pretty = ", ".join(f"{k}:{v}" for k, v in results.items())
        return f"MEASURE all → {pretty}", results

    def __repr__(self):
        return f"MEASURE -I {self.input_token}" if self.input_token else "MEASURE"


class CallASTNode:
    def __init__(self, tokens):
        if len(tokens) < 2:
            raise ValueError("CALL requires a process name")
        self.proc = tokens[1]
        U = [t.upper() for t in tokens]
        if "-I" in U:
            i = U.index("-I")
            self.params = tokens[i + 1 :]
        else:
            self.params = []

    def is_ready(self, *args):
        return True

    def evaluate(self, memory, current_cycle, qpu, hilbert, simulator=None):
        if simulator is None:
            raise ValueError("CALL requires a simulator")
        indent = "    " * simulator.subprocess_depth

        if self.proc not in simulator.compiled_processes:
            CompileASTNode(
                ["COMPILEPROCESS", "--NAME", self.proc, f"{self.proc}.txt"]
            ).evaluate(memory, current_cycle, qpu, hilbert, simulator)

        info = simulator.compiled_processes[self.proc]
        lines = list(info["lines"])

        if info["param_defs"] and self.params:
            assigns = assign_parameter_values(info["param_defs"], self.params)
            lines = substitute_parameters(lines, assigns)

        transcript = [f"{indent}=== Subprocess '{self.proc}' START ==="]
        if simulator.subprocess_depth == 0:
            transcript.append(
                f"{indent}Initial Memory:\n{generate_complete_memory_snapshot(simulator)}"
            )

        simulator.subprocess_depth += 1
        inner = "    " * simulator.subprocess_depth

        for idx, L in enumerate(lines, 1):
            t = L.strip()
            if not t or t.upper().startswith(("PARAMS:", "#", "/*")):
                continue
            if t.upper().startswith("CYCLE"):
                _, nstr = t.split()
                n = int(nstr)
                transcript.append(f"{inner}[Line {idx}] Advancing {n} cycles")
                for _ in range(n):
                    simulator.run_cycle(suppress_output=True)
                continue

            transcript.append(f"{inner}[Line {idx}] Executing: {t}")
            node = parse_command(t)
            try:
                msg, _ = node.evaluate(
                    memory, simulator.current_cycle, qpu, hilbert, simulator
                )
                transcript.append(f"{inner}  ↳ {msg}")
            except Exception as e:
                transcript.append(f"{inner}  ↳ ERROR: {e}")
                raise ValueError(f"Subprocess '{self.proc}' line {idx} failed: {e}")

        simulator.subprocess_depth -= 1
        transcript.append(f"{indent}=== Subprocess '{self.proc}' END ===")
        if simulator.subprocess_depth == 0:
            transcript.append(
                f"{indent}Final Memory:\n{generate_complete_memory_snapshot(simulator)}"
            )

        return "\n".join(transcript), None

    def __repr__(self):
        if self.params:
            return f"CALL {self.proc} -I " + " ".join(self.params)
        else:
            return f"CALL {self.proc}"


class JoinASTNode:
    def __init__(self, input_tokens, output_token):
        self.inputs = []
        for tok in input_tokens:
            q, c = tok.split(":")
            k = int(q) if q.isdigit() else q
            self.inputs.append((k, int(c)))

        if ":" in output_token:
            q, c = output_token.split(":")
            self.output = (int(q) if q.isdigit() else q, int(c))
        else:
            self.output = (output_token, None)

    def is_ready(self, memory, current_cycle, qpu=None):
        for k, cy in self.inputs:
            try:
                readdress_value(memory, k, cy, qpu)
            except:
                return False
        return True

    @staticmethod
    def join_registers(states):
        resolved = [resolve_state(s) for s in states]
        joined = resolved[0]
        for s in resolved[1:]:
            joined = np.kron(joined, s)
        expected = 2 ** len(resolved)
        if joined.shape[0] != expected:
            raise ValueError(
                f"JOIN: got dimension {joined.shape[0]}, expected {expected}"
            )
        return joined, expected

    def evaluate(self, memory, current_cycle, qpu, hilbert, simulator=None):
        states = [readdress_value(memory, k, cy, qpu) for k, cy in self.inputs]
        joined, _ = JoinASTNode.join_registers(states)
        out_k, out_c = self.output
        if out_c is None:
            out_c = current_cycle
        memory.setdefault(out_k, {})[out_c] = joined
        hilbert.output(out_k, out_c, joined)
        qpu._update_register(out_k, joined)
        return f"JOIN → {format_qubit_state(joined)}", joined

    def __repr__(self):
        ins = " ".join(f"{k}:{c}" for k, c in self.inputs)
        out, c = self.output
        return f"JOIN -I {ins} -O {out}:{c}"


class SplitASTNode:
    def __init__(self, tokens):
        if len(tokens) != 3:
            raise ValueError("SPLIT requires 3 arguments")
        self.input_ref, self.output_ref = tokens[0], tokens[1]
        try:
            self.target_dim = int(tokens[2])
        except:
            raise ValueError(f"Invalid target dim {tokens[2]}")

    def is_ready(self, memory, current_cycle, qpu=None):
        k, c = self.input_ref.split(":")
        k = int(k) if k.isdigit() else k
        c = int(c)
        return k in memory and c in memory[k] and c < current_cycle

    def evaluate(self, memory, current_cycle, qpu, hilbert, simulator=None):
        in_k, in_c = self.input_ref.split(":")
        out_k, out_c = self.output_ref.split(":")
        in_k = int(in_k) if in_k.isdigit() else in_k
        in_c = int(in_c)
        out_k = int(out_k) if out_k.isdigit() else out_k
        out_c = int(out_c)

        state = memory[in_k][in_c]
        if isinstance(state, tuple):
            state = state[1]

        new_state = qpu.split_register(in_k, state, self.target_dim)
        memory.setdefault(out_k, {})[out_c] = new_state
        hilbert.output(out_k, out_c, new_state)
        qpu._update_register(out_k, new_state)
        return f"SPLIT → {format_qubit_state(new_state)}", new_state

    def __repr__(self):
        return f"SPLIT {self.input_ref} {self.output_ref} {self.target_dim}"


# ------------------------------------------------------------------------
# Derived‐gate AST node: maps NOT, AND, NAND, OR, XOR → CCNOT building blocks
# ------------------------------------------------------------------------
class DerivedGateASTNode:
    """
    Handles:
      NOT  -I A:c    -O A:c   → CCNOT(A,1p,1p)
      AND  -I A:c B:c -O T:c  → CCNOT(A,B,0p); CNOT(0p→T)
      NAND -I A:c B:c -O T:c  → CCNOT(A,B,1p); CNOT(1p→T)
      OR   -I A:c B:c -O T:c  → NOT(A),NOT(B),AND(NOT(A),NOT(B)),CNOT(0p→T)
      XOR  -I A:c B:c -O T:c  → CCNOT(A,1p,T); CCNOT(B,T,T)
    """
    def __init__(self, gate, input_tokens, output_token):
        self.gate = gate.upper()
        self.inputs = []
        for tok in input_tokens:
            q, c = tok.split(":")
            idx = interpret_token(q)
            self.inputs.append((idx, int(c)))
        q_out, c_out = output_token.split(":")
        self.out = (interpret_token(q_out), int(c_out))

    def is_ready(self, memory, current_cycle, qpu=None):
        for key, cyc in self.inputs:
            try:
                readdress_value(memory, key, cyc, qpu)
            except:
                return False
        return True

    def evaluate(self, memory, current_cycle, qpu, hilbert, simulator=None):
        # ensure constants
        for const in ("1p", "0p"):
            if const not in qpu.custom_states:
                state = np.array([0,1],dtype=complex) if const=="1p" else np.array([1,0],dtype=complex)
                qpu.custom_states[const] = state
                if simulator: simulator.custom_tokens[const] = const

        out_q, out_c = self.out
        ops = []

        if self.gate == "NOT":
            (a,_), = self.inputs
            ops.append(("CCNOT", (a,"1p"), out_q))

        elif self.gate == "AND":
            (a,_),(b,_) = self.inputs
            anc = "0p"
            ops.append(("CCNOT", (a,b), anc))
            ops.append(("CNOT", (anc,), out_q))

        elif self.gate == "NAND":
            (a,_),(b,_) = self.inputs
            anc = "1p"
            ops.append(("CCNOT", (a,b), anc))
            ops.append(("CNOT", (anc,), out_q))

        elif self.gate == "OR":
            (a,_),(b,_) = self.inputs
            a_not = f"{a}_not"; b_not = f"{b}_not"
            for tmp in (a_not,b_not):
                if tmp not in qpu.custom_states:
                    qpu.custom_states[tmp] = np.array([1,0],dtype=complex)
            ops += [
                ("CCNOT", (a,"1p"), a_not),
                ("CCNOT", (b,"1p"), b_not),
                ("CCNOT", (a_not,b_not), "0p"),
                ("CNOT", ("0p",), out_q),
            ]

        elif self.gate == "XOR":
            (a,_),(b,_) = self.inputs
            tmp = f"{out_q}_tmp"
            if tmp not in qpu.custom_states:
                qpu.custom_states[tmp] = np.array([1,0],dtype=complex)
            ops += [
                ("CCNOT", (a,"1p"), tmp),
                ("CCNOT", (b,tmp),  out_q),
            ]

        else:
            raise NotImplementedError(f"Derived gate {self.gate} not supported")

        # execute
        for op, ctrls, tgt in ops:
            if op == "CNOT":
                qpu.apply_controlled_gate(X, ctrls[0], tgt)
            else:  # CCNOT
                qpu.apply_ccnot(ctrls[0], ctrls[1], tgt)

        final = qpu.local_states[out_q]
        memory.setdefault(out_q, {})[out_c] = final
        hilbert.output(out_q, out_c, final)
        return f"{self.gate} → {out_q}", final

    def __repr__(self):
        ins = " ".join(f"{k}:{c}" for k,c in self.inputs)
        o,c   = self.out
        return f"{self.gate} -I {ins} -O {o}:{c}"


class AndASTNode:
    def __init__(self, input_tokens, output_token):
        if len(input_tokens) != 2:
            raise ValueError("AND requires exactly 2 inputs")
        self.inputs = []
        for tok in input_tokens:
            q, c = tok.split(":")
            idx = interpret_token(q)
            self.inputs.append((idx, int(c)))
        q_out, c_out = output_token.split(":")
        self.output = (interpret_token(q_out), int(c_out))

    def is_ready(self, memory, current_cycle, qpu=None):
        for key, cy in self.inputs:
            try:
                readdress_value(memory, key, cy, qpu)
            except:
                return False
        return True

    def evaluate(self, memory, current_cycle, qpu, hilbert, simulator=None):
        (ctrl1,_),(ctrl2,_) = self.inputs
        tgt, tgt_cycle = self.output
        new_state = qpu.apply_ccnot(ctrl1, ctrl2, tgt)
        memory.setdefault(tgt, {})[tgt_cycle] = new_state
        hilbert.output(tgt, tgt_cycle, new_state)
        return f"AND {ctrl1},{ctrl2} → {tgt}", new_state

    def __repr__(self):
        ins = " ".join(f"{k}:{c}" for k,c in self.inputs)
        o,c = self.output
        return f"AND -I {ins} -O {o}:{c}"


class OrASTNode:
    def __init__(self, input_tokens, output_token):
        if len(input_tokens) != 2:
            raise ValueError("OR requires exactly 2 inputs")
        self.inputs = []
        for tok in input_tokens:
            q,c = tok.split(":")
            idx = interpret_token(q)
            self.inputs.append((idx,int(c)))
        q_out, c_out = output_token.split(":")
        self.output = (interpret_token(q_out), int(c_out))
        self.target_qubit = self.output[0]

    def is_ready(self, memory, current_cycle, qpu=None):
        for k,cy in self.inputs:
            try:
                readdress_value(memory, k, cy, qpu)
            except:
                return False
        return True

    def evaluate(self, memory, current_cycle, qpu, hilbert, simulator=None):
        ctrl1,_ = self.inputs[0]
        ctrl2,_ = self.inputs[1]
        qpu.apply_single_qubit_gate(X, ctrl1)
        qpu.apply_single_qubit_gatee(X, ctrl2)
        qpu.apply_ccnot(ctrl1, ctrl2, self.target_qubit)
        qpu.apply_single_qubit_gate(X, ctrl1)
        qpu.apply_single_qubit_gate(X, ctrl2)
        qpu.apply_single_qubit_gate(X, self.target_qubit)
        new_state = qpu.local_states[self.target_qubit]
        out_k, out_c = self.output
        memory.setdefault(out_k, {})[out_c] = new_state
        hilbert.output(out_k, out_c, new_state)
        return f"OR {ctrl1},{ctrl2} → {out_k}", new_state

    def __repr__(self):
        ins = " ".join(f"{k}:{c}" for k,c in self.inputs)
        o,c = self.output
        return f"OR -I {ins} -O {o}:{c}"


class ReturnValsASTNode:
    """AST node for RETURNVALS <v1> <v2> ... inside a child process."""
    def __init__(self, tokens):
        if len(tokens) < 2:
            raise ValueError("RETURNVALS requires at least one token")
        self.vals = tokens[1:]

    def is_ready(self, *args):
        return True

    def evaluate(self, memory, current_cycle, qpu, hilbert, simulator=None):
        simulator.child_return_keys = self.vals
        return f"Child will return {self.vals}", None

    def __repr__(self):
        return "RETURNVALS " + " ".join(self.vals)


class AcceptValsASTNode:
    """AST node for ACCEPTVALS <l1> <l2> ... in the parent, to capture child returns."""
    def __init__(self, tokens):
        if len(tokens) < 2:
            raise ValueError("ACCEPTVALS requires at least one token")
        self.locals = tokens[1:]

    def is_ready(self, *args):
        return True

    def evaluate(self, memory, current_cycle, qpu, hilbert, simulator=None):
        mapping = simulator.last_child_returns
        for lk, rv in zip(self.locals, mapping.values()):
            memory.setdefault(lk, {})[current_cycle] = rv
        return f"Accepted {list(mapping.keys())} → {self.locals}", None

    def __repr__(self):
        return "ACCEPTVALS " + " ".join(self.locals)


# ------------------------------------------------------------------------
# Generic gate AST node for primitive gates
# ------------------------------------------------------------------------
class GateASTNode:
    def __init__(self, gate_token, input_tokens, output_token):
        if gate_token.upper().startswith("B"):
            self.reverse = True
            gate_token = gate_token[1:]
        else:
            self.reverse = False

        if "=" in gate_token:
            g, p = gate_token.split("=", 1)
            self.gate = g.upper()
            self.param = float(p)
        else:
            self.gate = gate_token.upper()
            self.param = None

        if self.reverse and self.gate == "PHASE" and self.param is not None:
            self.param = -self.param

        self.inputs = []
        for tok in input_tokens:
            if ":" in tok:
                q, c = tok.split(":", 1)
                idx = interpret_token(q)
                cyc = int(c)
            else:
                idx = interpret_token(tok)
                cyc = None
            self.inputs.append((idx, cyc))

        if ":" in output_token:
            q, c = output_token.split(":", 1)
            out_idx = int(q) if q.isdigit() else q
            out_cyc = int(c)
        else:
            out_idx = interpret_token(output_token)
            out_cyc = None
        self.output = (out_idx, out_cyc)

    def is_ready(self, memory, current_cycle, qpu=None):
        for idx, cyc in self.inputs:
            use_c = cyc if cyc is not None else current_cycle
            try:
                readdress_value(memory, idx, use_c, qpu)
            except:
                return False
        return True

    def evaluate(self, memory, current_cycle, qpu, hilbert, simulator=None):
        out_k, out_c = self.output

        # --- fast‐path CNOT ---
        if self.gate == "CNOT" and isinstance(out_k, int):
            # let the QPU do the flip exactly once
            ctrl, _ = self.inputs[0]
            new = qpu.apply_controlled_gate(X, ctrl, out_k)
            # record the post‐gate state
            cyc = out_c if out_c is not None else current_cycle
            memory.setdefault(out_k, {})[cyc] = new
            hilbert.output(out_k, cyc, new)
            return f"CNOT {ctrl}→{out_k}", new

        # --- fast‐path CCNOT (Toffoli) ---
        if self.gate == "CCNOT" and isinstance(out_k, int):
            c1, _ = self.inputs[0]
            c2, _ = self.inputs[1]
            new = qpu.apply_ccnot(c1, c2, out_k)
            cyc = out_c if out_c is not None else current_cycle
            memory.setdefault(out_k, {})[cyc] = new
            hilbert.output(out_k, cyc, new)
            return f"CCNOT {c1},{c2}→{out_k}", new

        # --- fallback to the generic gate‐handler path ---
        # ensure any custom‐token registers exist
        if isinstance(out_k, str) and out_k not in qpu.custom_states:
            qpu.custom_states[out_k] = np.array([1, 0], dtype=complex)
            if simulator:
                simulator.custom_tokens[out_k] = out_k
        for idx, _ in self.inputs:
            if isinstance(idx, str) and idx not in qpu.custom_states:
                qpu.custom_states[idx] = np.array([1, 0], dtype=complex)
                if simulator:
                    simulator.custom_tokens[idx] = idx

        info = gate_handlers.get(self.gate)
        if info is None:
            raise NotImplementedError(f"Gate {self.gate} not implemented")
        if len(self.inputs) != info["inputs"]:
            raise ValueError(f"Gate {self.gate} expects {info['inputs']} inputs")

        ins = [i for i, _ in self.inputs]
        msg, comp = info["handler"](qpu, ins, out_k, self.param)
        cyc = out_c or current_cycle
        memory.setdefault(out_k, {})[cyc] = comp
        hilbert.output(out_k, cyc, comp)
        return msg, comp


    def __repr__(self):
        ins = []
        for i, c in self.inputs:
            ins.append(f"{i}:{c}" if c is not None else f"{i}")
        o, c = self.output
        pc = f":{c}" if c is not None else ""
        p = f"={self.param}" if self.param is not None else ""
        rev = " (B)" if self.reverse else ""
        return f"{self.gate}{p}{rev} -I {' '.join(ins)} -O {o}{pc}"


# ------------------------------------------------------------------------
# Parser entrypoint
# ------------------------------------------------------------------------
def parse_command(command_str: str):
    tokens = command_str.strip().split()
    if not tokens:
        raise ValueError("Empty command")

    # merge PHASE= syntax
    first = tokens[0]
    if first.endswith("=") and first[:-1].upper() in gate_handlers:
        if len(tokens) < 2:
            raise ValueError(f"Missing parameter after '{first}'")
        merged = first + tokens[1]
        tokens = [merged] + tokens[2:]

    cmd = tokens[0].upper()
    upp = [t.upper() for t in tokens]

    if cmd == "CYCLE":
        return CycleASTNode(tokens)
    if cmd == "COMPILEPROCESS":
        return CompileASTNode(tokens)
    if cmd == "FREE":
        if "-I" not in upp:
            raise ValueError("FREE requires -I")
        return FreeASTNode(tokens[upp.index("-I")+1:])
    if cmd == "SET":
        return SetASTNode(tokens[1:])
    if cmd == "JOIN":
        if "-I" not in upp or "-O" not in upp:
            raise ValueError("JOIN requires -I and -O")
        i_idx, o_idx = upp.index("-I"), upp.index("-O")
        return JoinASTNode(tokens[i_idx+1:o_idx], tokens[o_idx+1])
    if cmd == "SPLIT":
        return SplitASTNode(tokens[1:])
    if cmd == "CALL":
        return CallASTNode(tokens)
    if cmd == "CREATETOKEN":
        return CreateTokenASTNode(tokens[1:])
    if cmd == "DELETETOKEN":
        return DeleteTokenASTNode(tokens[1:])
    if cmd == "AND":
        if "-I" not in upp or "-O" not in upp:
            raise ValueError("AND requires -I and -O")
        i_idx, o_idx = upp.index("-I"), upp.index("-O")
        return DerivedGateASTNode("AND", tokens[i_idx+1:o_idx], tokens[o_idx+1])
    if cmd == "NAND":
        if "-I" not in upp or "-O" not in upp:
            raise ValueError("NAND requires -I and -O")
        i_idx, o_idx = upp.index("-I"), upp.index("-O")
        return DerivedGateASTNode("NAND", tokens[i_idx+1:o_idx], tokens[o_idx+1])
    if cmd == "OR":
        if "-I" not in upp or "-O" not in upp:
            raise ValueError("OR requires -I and -O")
        i_idx, o_idx = upp.index("-I"), upp.index("-O")
        return DerivedGateASTNode("OR", tokens[i_idx+1:o_idx], tokens[o_idx+1])
    if cmd == "NOT":
        if "-I" not in upp or "-O" not in upp:
            raise ValueError("NOT requires -I and -O")
        i_idx, o_idx = upp.index("-I"), upp.index("-O")
        return DerivedGateASTNode("NOT", tokens[i_idx+1:o_idx], tokens[o_idx+1])
    if cmd == "XOR":
        if "-I" not in upp or "-O" not in upp:
            raise ValueError("XOR requires -I and -O")
        i_idx, o_idx = upp.index("-I"), upp.index("-O")
        return DerivedGateASTNode("XOR", tokens[i_idx+1:o_idx], tokens[o_idx+1])
    if cmd == "MEASURE":
        if "-I" in upp:
            i_idx = upp.index("-I")
            return MeasureASTNode(tokens[i_idx+1])
        return MeasureASTNode(None)
    if cmd == "RETURNVALS":
        return ReturnValsASTNode(tokens)
    if cmd == "ACCEPTVALS":
        return AcceptValsASTNode(tokens)
    if cmd == "MAIN-PROCESS":
        return MainProcessASTNode(tokens)
    if cmd == "CREATETOKEN":
        if "-I" not in upp:
            raise ValueError("CREATETOKEN requires -I")
        i_idx = upp.index("-I")
        return CreateTokenASTNode(tokens[i_idx + 1:])
    if cmd == "DELETETOKEN":
        if "-I" not in upp:
            raise ValueError("DELETETOKEN requires -I")
        i_idx = upp.index("-I")
        return DeleteTokenASTNode(tokens[i_idx + 1:])

    base = cmd.split("=",1)[0]
    if base in gate_handlers:
        if "-I" not in upp or "-O" not in upp:
            raise ValueError(f"Gate {base} requires -I and -O")
        i_idx, o_idx = upp.index("-I"), upp.index("-O")
        return GateASTNode(tokens[0], tokens[i_idx+1:o_idx], tokens[o_idx+1])


    raise ValueError(f"Unknown command: {cmd}")
