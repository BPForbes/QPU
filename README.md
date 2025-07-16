# Universal Quantum Processor

This repository implements a simple quantum circuit simulator and a custom scripting language.  The syntax is inspired by assembly style instructions and is parsed by `parse_command` in `qpu/ast.py`.


## Instruction Set

The scripting language exposes compiler-like instructions that are queued and executed cycle by cycle. Key commands include:

- `SET <addr> <value>` – assign a register.
- `INCREASECYCLE` – increment the current cycle.
- `COMPILEPROCESS --NAME <proc> <file>` – compile a process definition from a text file.
- `CALL <proc> -I <args>` – invoke a compiled process.
- `MAIN-PROCESS <name>` – mark the main routine.
- `DECLARECHILD <name>` / `RUNCHILD <name> -I <args>` – manage sub‑processes.
- `RETURNVALS <keys>` and `ACCEPTVALS <locals>` – exchange data with a parent process.
- `CREATETOKEN -I <name...>` / `DELETETOKEN -I <name...>` – manage custom qubit tokens.
- `FREE -I <k:c ...>` – release memory entries.
- `JOIN -I <A:c B:c ...> -O <Q:c>` – combine registers.
- `SPLIT <A:c> <B:c> <dim>` – split a register into `dim` parts.
- `MEASURE` or `MEASURE -I <q[:c]>` – collapse qubits.
- `SAVE_STATE <label>` and `LOAD_STATE <label>` – checkpoint and restore.
- Derived boolean gates (`AND`, `NAND`, `OR`, `NOT`, `XOR`) built from quantum primitives.
- Primitive gates `X`, `H`, `CNOT`, `CCNOT`, `PHASE=<θ>` with `-I` and `-O` arguments.


Arguments typically use the form `qubit:cycle`, where the cycle component is optional. The interpreter re‑addresses values according to the specified cycle.

## AST Overview

Each command maps to an AST node in `qpu/ast.py`. During simulation every node checks `is_ready` and, once ready, `evaluate` executes it. Key node types include:

- `SetASTNode` for `SET` commands.
- `CycleASTNode` to advance the clock.
- `CompileASTNode` and `CallASTNode` for process management.
- `MainProcessASTNode` marks the entry routine.
- `DeclareChildASTNode` and `RunChildASTNode` for spawning subprocesses.
- `CreateTokenASTNode` and `DeleteTokenASTNode` to manage custom registers.
- `JoinASTNode` and `SplitASTNode` for register manipulation.
- `FreeASTNode` to clear memory.
- `MeasureASTNode` collapses qubits.
- `ReturnValsASTNode` and `AcceptValsASTNode` exchange values with a parent.
- `SaveStateASTNode` and `LoadStateASTNode` handle checkpoints.
- `GateASTNode` and `DerivedGateASTNode` implement primitive and boolean gates.

The simulator advances by repeatedly calling `INCREASECYCLE`; there is no multi-step `CYCLE n` instruction.

### Example

```
SET 0:0 1p
H -I 0:0 -O 0:0
CNOT -I 0:0 -O 1:0
MEASURE -I 1
```

## Process Compilation and Parameters

Processes live in plain text files. Use `COMPILEPROCESS` to parse and store a
process, then `RUNPROCESS` to execute it. The optional `PARAMS:` header lists
arguments in order. Types include `state` (token of a qubit) and `int`.
Missing arguments default to `0p` or `0` depending on the type.  Placeholders
are written as `$Name` in the script.

```
PARAMS: A0:state A1:state Cin:state Sum:int
MAIN-PROCESS Example
SET A0:0 $A0
SET A1:0 $A1
INCREASECYCLE
SET Cin:0 $Cin
RETURNVALS Sum
```

Compile and run via the CLI:

```
COMPILEPROCESS --NAME Example Example.txt
RUNPROCESS --NAME Example 1p 0p 0p 0
```

## Gate Reference

Primitive single-qubit gates and multi-qubit controls are listed below along with their matrices.  Tables describe how each gate acts on computational basis states.

### Identity (I)

```math
I = \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}
```

This gate leaves the qubit unchanged.

### Pauli‑X (NOT)

```math
X = \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix}
```

| q | X(q) |
|---|------|
| 0 | 1 |
| 1 | 0 |

### Pauli‑Y

```math
Y = \begin{bmatrix}0 & -i \\ i & 0\end{bmatrix}
```

| q | Y(q) |
|---|------|
| $|0\rangle$ | $i|1\rangle$ |
| $|1\rangle$ | $-i|0\rangle$ |

### Pauli‑Z

```math
Z = \begin{bmatrix}1 & 0 \\ 0 & -1\end{bmatrix}
```

| q | Z(q) |
|---|------|
| $|0\rangle$ | $|0\rangle$ |
| $|1\rangle$ | $-|1\rangle$ |

### Controlled‑NOT (CNOT)

The 4×4 matrix for CNOT is

```math
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0
\end{bmatrix}
```

| A (control) | B (target in) | B (out) |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

### Toffoli (CCNOT)

```math
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
\end{bmatrix}
```

| A | B | T (in) | T (out) |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 0 | 0 | 1 | 1 |
| 0 | 1 | 0 | 0 |
| 0 | 1 | 1 | 1 |
| 1 | 0 | 0 | 0 |
| 1 | 0 | 1 | 1 |
| 1 | 1 | 0 | 1 |
| 1 | 1 | 1 | 0 |

### Hadamard (H)

```math
H = \frac{1}{\sqrt{2}}\begin{bmatrix}1 & 1 \\ 1 & -1\end{bmatrix}
```

| q | H(q) |
|---|---|
| $|0\rangle$ | $(|0\rangle + |1\rangle)/\sqrt{2}$ |
| $|1\rangle$ | $(|0\rangle - |1\rangle)/\sqrt{2}$ |

### PHASE(θ)

```math
\begin{bmatrix}
1 & 0 \\
0 & e^{i\theta}
\end{bmatrix}
```

| q | P\_θ(q) |
|---|---|
| $|0\rangle$ | $|0\rangle$ |
| $|1\rangle$ | $e^{i\theta}|1\rangle$ |

### Boolean Gates via CCNOT / CNOT

| A | B | AND | NAND | OR | XOR |
|---|---|-----|------|----|-----|
| 0 | 0 | 0   | 1    | 0  | 0   |
| 0 | 1 | 0   | 1    | 1  | 1   |
| 1 | 0 | 0   | 1    | 1  | 1   |
| 1 | 1 | 1   | 0    | 1  | 0   |

These boolean primitives are translated internally into combinations of CNOT and CCNOT operations defined in `qpu/qpu_base.py`.

## Backend Logic

`QuantumProcessorUnit` manages qubit registers and applies gates.  Each physical or custom qubit is stored as a state vector $|\psi\rangle \in \mathbb{C}^2$.  Multi‑qubit registers are represented by Kronecker products.  Measurement collapses a register to a computational basis state based on the probability amplitudes $|a|^2$.

The simulator (`CircuitSimulator`) maintains a clock and a queue of AST nodes.  During each cycle it executes any node whose input dependencies are satisfied.  Outputs are logged to a `HilbertSpace` object for inspection or debugging.

Processes written in the scripting language can be compiled from plain text using `COMPILEPROCESS`.  The compiler supports parameter substitution using the `PARAMS:` header.  Compiled processes can be called directly or run as child processes, with return values exchanged via `RETURNVALS` and `ACCEPTVALS`.

## Running

Execute all unit tests:

```bash
python unittests.py
```

Start the interactive CLI:

```bash
python main.py
```

