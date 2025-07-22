# cli.py

"""
Interactive CLI for the Universal Quantum Circuit Simulator,
now with parent↔child process support (RUNCHILD, RETURNVALS, ACCEPTVALS).
"""

from qpu.ast import (
    parse_command,
    generate_complete_memory_snapshot,
    binary_to_ascii,
    # for run_child:
    assign_parameter_values,
    substitute_parameters,
    readdress_value,
    ClockFrame,
)
from dataclasses import dataclass
from qpu.hilbert import HilbertSpace
from qpu.qpu_base import QuantumProcessorUnit, format_qubit_state
import numpy as np

# ANSI colors
COLORS = {
    "reset": "\033[0m",
    "red":   "\033[31m",
    "green": "\033[32m",
    "yellow":"\033[33m",
    "blue":  "\033[34m",
    "magenta":"\033[35m",
    "cyan":  "\033[36m"
}

def color_text(text, color):
    return f"{COLORS.get(color,'reset')}{text}{COLORS['reset']}"

def print_help():
    print(f"""
{color_text('=== Universal Quantum CLI ===','yellow')}

{color_text('Compilation & Run','cyan')}
  COMPILEPROCESS --NAME <proc> <file>    # auto‑compiles CALL‑ed deps
  RUNPROCESS --NAME <proc>               # runs the compiled process

{color_text('AST Commands','cyan')}
  <GATE> -I <q>:<c> [...] -O <q>:<c>
  SET    <q>:<c> <value>
  JOIN   -I <q>:<c> [...] -O <q>:<c>
  FREE   -I <q>:<c> [...]
  CALL   <Process> -I <params> [...]

{color_text('Child Process','cyan')}
  MAIN-PROCESS <name>
  DECLARECHILD <child>
  RUNCHILD <child> -I <args> -O <rets>
  RETURNVALS <keys…>
  ACCEPTVALS <locals…>

{color_text('Other','cyan')}
  RUNTIMEFP    Runtime history fingerprint
  FPRINTMEM    Full memory snapshot
  FINGERPRINTMEMORY  Full runtime report + fingerprint
  QPUFPRINT    QPU T1/T2 fingerprint
  SIGNFPRINT   Sign runtime fingerprint via XMSS
  BB84         BB84 key exchange demo
  HELP, EXIT
""")

class CircuitSimulator:
    def __init__(self, qpu: QuantumProcessorUnit):
        self.qpu = qpu
        self.memory = {}
        self.ast_nodes = []
        self.hilbert = HilbertSpace()
        self.max_cycles = qpu.num_qubits + 1

        self._clock_stack = [ClockFrame("__main__", base=0, local=0)]
        self.checkpoints = {}

        # compiled processes registry
        self.compiled_processes = {}

        # custom tokens in use
        self.custom_tokens = {}

        # for sub‑process nesting
        self.subprocess_depth = 0

        # history for runtime fingerprinting
        self.runtime_history = []
        # per-process stack of SET locks
        self._lock_stack = [set()]

        # value set by MASTERVAL
        self.master_value = None

        # child‑process support
        self.child_return_keys = []
        self.last_child_returns = {}

    # Clock helpers
    @property
    def current_cycle(self):
        frame = self._clock_stack[-1]
        return frame.base + frame.local

    @current_cycle.setter
    def current_cycle(self, value):
        frame = self._clock_stack[-1]
        frame.local = value - frame.base

    def increase_cycle(self):
        self._clock_stack[-1].local += 1
        if hasattr(self.qpu, "apply_idle_noise"):
            self.qpu.apply_idle_noise()

    def prune_cycles(self):
        for k in list(self.memory):
            self.memory[k] = {c: v for c, v in self.memory[k].items()
                              if c <= self.max_cycles}
            if not self.memory[k]:
                del self.memory[k]
        self.hilbert.prune(self.max_cycles)

    def run_cycle(self, suppress_output=False):
        if not suppress_output:
            print(color_text(f"\n--- Cycle {self.current_cycle} ---", "cyan"))
            print(color_text("Memory:", "magenta"))
            print(generate_complete_memory_snapshot(self))
        for node in list(self.ast_nodes):
            if node.is_ready(self.memory, self.current_cycle, self.qpu):
                try:
                    msg, _ = node.evaluate(
                        self.memory,
                        self.current_cycle,
                        self.qpu,
                        self.hilbert,
                        self
                    )
                    if not suppress_output:
                        print(color_text(f"✔ {node}: {msg}", "green"))
                except Exception as e:
                    if not suppress_output:
                        print(color_text(f"✗ {node}: {e}", "red"))
        self.increase_cycle()

    def free_memory_entries(self, entries):
        msgs = []
        for k, c in entries:
            if k in self.memory and c in self.memory[k]:
                del self.memory[k][c]
                msgs.append(f"{k}@{c} freed")
        return ", ".join(msgs)

    # ——— child‑process support —————————————————————————————————————
    def lock_id(self, key):
        self._lock_stack[-1].add(key)

    def unlock_id(self, key):
        self._lock_stack[-1].discard(key)

    def is_locked(self, key):
        return key in self._lock_stack[-1]

    def start_process(self):
        self._lock_stack.append(set())

    def end_process(self):
        if len(self._lock_stack) > 1:
            to_unlock = self._lock_stack.pop()

            # reset each unlocked ID to |0> at the current cycle
            for qid in to_unlock:
                zero_state = np.array([1.0, 0.0], dtype=complex)
                # update QPU internal state
                self.qpu.custom_states[qid] = zero_state
                # record in memory and Hilbert output
                cyc = self.current_cycle
                self.memory.setdefault(qid, {})[cyc] = zero_state
                self.hilbert.output(qid, cyc, zero_state)
                # remove from custom_tokens so it can be reallocated
                self.custom_tokens.pop(qid, None)

    def clear_locks(self):
        self._lock_stack[-1].clear()

    def get_runtime_history_string(self):
        return "\n".join(self.runtime_history)

    def get_runtime_fingerprint(self):
        hist = self.get_runtime_history_string().encode()
        bp = ''.join(format(b, '08b') for b in hist)
        return binary_to_ascii(bp), self.get_runtime_history_string()

    def process_summary(self):
        rep = "\n".join(self.runtime_history)
        rep += "\nFinal Memory:\n" + generate_complete_memory_snapshot(self)
        bp = ''.join(format(ord(c), '08b') for c in rep)
        return rep, binary_to_ascii(bp)

    def declare_child(self, child_name):
        if child_name not in self.compiled_processes:
            raise ValueError(f"Child '{child_name}' not compiled")

    def run_child(self, name, args, parent_tokens):
        if name not in self.compiled_processes:
            raise ValueError(f"Child '{name}' not compiled")

        info = self.compiled_processes[name]
        sections = info["sections"]
        lines = (
            sections["child_prologue"]
            + sections["body"]
            + sections["children_wrapup"]
        )

        if info.get("param_defs") and args:
            assigns = assign_parameter_values(info["param_defs"], args)
            lines = substitute_parameters(lines, assigns)

        injected_tokens = []
        original_states = {}
        for token in parent_tokens.values():
            if token not in self.qpu.custom_states:
                self.qpu.custom_states[token] = np.array([1, 0], dtype=complex)
                injected_tokens.append(token)
            else:
                original_states[token] = self.qpu.custom_states[token].copy()
            if token not in self.custom_tokens:
                self.custom_tokens[token] = token

        start_global = self.current_cycle
        self._clock_stack.append(ClockFrame(name, base=start_global, local=0))
        self.start_process()

        for L in lines:
            t = L.strip()
            if not t or t.upper().startswith(("PARAMS:", "#", "/*")):
                continue
            node = parse_command(t)
            node.evaluate(self.memory, self.current_cycle, self.qpu, self.hilbert, self)
            self.increase_cycle()

        ret_vals = []
        from qpu.ast import interpret_token
        final_cycle = self.current_cycle - 1
        for key in self.child_return_keys:
            k = interpret_token(key)
            val = readdress_value(self.memory, k, final_cycle, self.qpu)
            ret_vals.append(val)

        updated_tokens = {}
        for token in parent_tokens.values():
            if token in self.qpu.custom_states:
                new_state = self.qpu.custom_states[token]
                old_state = original_states.get(token)
                if old_state is None or not np.array_equal(new_state, old_state):
                    updated_tokens[token] = format_qubit_state(new_state)

        for token in injected_tokens:
            if token in self.qpu.custom_states:
                del self.qpu.custom_states[token]
            if token in self.custom_tokens:
                del self.custom_tokens[token]

        self._clock_stack.pop()
        self.end_process()

        return ret_vals, updated_tokens

# ——— single‑process runner —————————————————————————————————————————
def run_single_process(cmd_line, sim: CircuitSimulator):
    parts = cmd_line.strip().split()
    U = [p.upper() for p in parts]
    if "--NAME" not in U:
        print(color_text("RUNPROCESS needs --NAME", "red"))
        return
    idx = U.index("--NAME")
    name = parts[idx + 1]
    args = parts[idx + 2:]
    if name not in sim.compiled_processes:
        print(color_text(f"Process '{name}' not compiled", "red"))
        return

    info = sim.compiled_processes[name]
    sections = info["sections"]
    lines = (
        sections["global_prologue"]
        + sections["main_prologue"]
        + sections["body"]
        + sections["main_wrapup"]
    )

    if info.get("param_defs"):
        assigns = assign_parameter_values(info["param_defs"], args)
        lines = substitute_parameters(lines, assigns)

    print(color_text(f"--- Process '{name}' START ---", "magenta"))
    sim.start_process()

    for L in lines:
        t = L.strip()
        if not t or t.upper().startswith(("PARAMS:", "#", "/*")):
            continue

        print(color_text(f"[Process] {t}", "blue"))
        node = parse_command(t)
        try:
            node.evaluate(sim.memory, sim.current_cycle, sim.qpu, sim.hilbert, sim)
            print(color_text("  ↳ OK", "green"))
        except Exception as e:
            print(color_text(f"  ↳ ERROR: {e}", "red"))

    sim.end_process()
    print(color_text(f"--- Process '{name}' END ---", "magenta"))
    print(color_text("Final Memory:", "yellow"))
    print(generate_complete_memory_snapshot(sim))

# ——— interactive loop —————————————————————————————————————————————
def interactive_cli():
    qpu = QuantumProcessorUnit(16)
    sim = CircuitSimulator(qpu)

    print(color_text("Welcome to the Universal Quantum CLI!", "green"))
    print_help()

    while True:
        try:
            inp = input(color_text(">> ", "yellow")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not inp:
            continue

        cmd = inp.split()[0].upper()
        if cmd == "EXIT":
            break
        if cmd == "HELP":
            print_help()
            continue

        if cmd == "COMPILEPROCESS":
            try:
                node = parse_command(inp)
                msg, _ = node.evaluate(sim.memory, sim.current_cycle, sim.qpu, sim.hilbert, sim)
                print(color_text(msg, "green"))
            except Exception as e:
                print(color_text(f"Error: {e}", "red"))
            continue

        if cmd == "RUNPROCESS":
            run_single_process(inp, sim)
            continue

        # Otherwise treat as an AST command
        try:
            node = parse_command(inp)
            sim.ast_nodes.append(node)
            print(color_text(f"Queued: {node}", "cyan"))
        except Exception as e:
            print(color_text(f"Parse error: {e}", "red"))

if __name__ == "__main__":
    interactive_cli()
