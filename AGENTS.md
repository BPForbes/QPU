## Summary
This `AGENTS.md` describes the QPU simulator’s custom AST-based syntax and gives Codex clear mappings from text commands to AST node classes. It covers general quantum concepts (qubits, superposition, entanglement, measurement), the overall parser workflow, and the precise syntax for each supported instruction—primitive gates, derived gates, memory operations, subprocess controls, and parameter handling—so that Codex can generate, complete, and refactor `.txt`-based quantum programs correctly.

## General Quantum Theory
- **Qubit**: Fundamental unit of quantum information. Unlike a classical bit, a qubit can exist in a superposition \|\Ψ⟩ = α|0⟩ + β|1⟩ with |α|² + |β|² = 1.  
- **Superposition**: A qubit in superposition holds both basis states simultaneously until measured.  
- **Entanglement**: Non-local correlation between qubits; operations on one qubit can instantaneously affect its entangled partners.  
- **Measurement**: Collapses a qubit’s state to |0⟩ or |1⟩ with probabilities given by its amplitudes (Born rule).  

## AST Overview
All commands in `.txt` instruction files are parsed by `parse_command(...)` in **qpu/ast.py**, which returns instances of AST node classes. Each AST node implements:  
- **is_ready(...)**: whether its inputs (memory or qubit state) are available,  
- **evaluate(...)**: applies the operation to the `QuantumProcessorUnit`, updates simulator memory and Hilbert outputs,  
- **`__repr__`**: for human-readable logging and debugging.  

The overall parser workflow:
1. **`read_protocol_file(filename)`** handles line continuations (`\`), C-style comments (`/*…*/`), shell-style comments (`#`), and blank lines.  
2. **Parameter handling**:  
   - **`parse_parameters(def_line)`** reads the `PARAMS:` definition line at the top of a file,  
   - **`assign_parameter_values(...)`** binds user-supplied values to those parameters,  
   - **`substitute_parameters(...)`** replaces all `$param` tokens through the protocol.  
3. **`parse_command(command_str)`** splits the cleaned line into tokens, matches the first token to one of the AST node types, and constructs the appropriate AST node instance.  

## Command Syntax

### Control & Process Nodes
- **`CYCLE n`**  
  Advance the simulator by `n` full clock cycles.  
  → `CycleASTNode`  
- **`MAIN-PROCESS name`**  
  Identify and switch to the main execution context.  
  → `MainProcessASTNode`  
- **`COMPILEPROCESS --NAME proc file.txt [-I p1 p2 …] [-$R]`**  
  Parse and fingerprint `file.txt` as a subprocess named `proc`, optionally substituting `PARAMS:` values or marking “no-params” with `-$R`.  
  → `CompileASTNode`  
- **`CALL proc [-I p1 p2 …]`**  
  Invoke a previously compiled subprocess, passing in parameters if needed. Recursively compiles missing dependencies.  
  → `CallASTNode`  
- **`RETURNVALS v1 v2 …`** / **`ACCEPTVALS l1 l2 …`**  
  Inside a child: declare which registers to return; in the parent: bind returned values to local names.  
  → `ReturnValsASTNode` / `AcceptValsASTNode`  

### Memory & Token Nodes
- **`SET q:cycle val`**  
  Initialize or overwrite register `q` at cycle `cycle` to one of:  
  - `0p` or `1p` (explicit |0⟩/|1⟩ states),  
  - `sp[_dimK]` (random superposition, optional dimension `K`).  
  → `SetASTNode`  
- **`FREE -I q:cycle [r:c …]`**  
  Release stored snapshots for the listed register@cycle entries.  
  → `FreeASTNode`  
- **`CREATETOKEN -I name1 [name2 …]`** / **`DELETETOKEN -I name1 [name2 …]`**  
  Dynamically allocate or remove custom 1-qubit registers (initialized to |0⟩).  
  → `CreateTokenASTNode` / `DeleteTokenASTNode`  

### Join & Split
- **`JOIN -I q1:c1 q2:c2 … -O out:c`**  
  Tensor-product join of multiple registers into one larger register, stored at `out:c`.  
  → `JoinASTNode`  
- **`SPLIT in:c out:c′ K`**  
  Partition an N-qubit register at `in:c` into a K-qubit register at `out:c′`, discarding the remainder.  
  → `SplitASTNode`  

### Measurement
- **`MEASURE`**  
  Collapse and record all registers at the current cycle.  
- **`MEASURE -I q[:cycle]`**  
  Collapse and record only `q` (at optional cycle).  
  → `MeasureASTNode`  

### Primitive Gates & Phase
Handled by `GateASTNode` or optimized in `gate_handlers`:
```bash
H        -I q[:c] -O out[:c′]   # Hadamard  
X, Y, Z  -I q[:c] -O out[:c′]   # Pauli gates  
CNOT/CX  -I ctrl[:c] -O tgt     # Controlled-NOT  
CCNOT/CCX                    # Toffoli  
PHASE=θ  -I q[:c] -O out[:c′]   # Z-rotation by θ radians  
````

* **`-I`** lists input qubit tokens; **`-O`** specifies the target/output qubit.
* Numeric tokens (e.g. `0`, `1`) refer to physical qubits; string tokens remain custom registers.

### Derived Gates

Convenience macros via `DerivedGateASTNode`:

```bash
NOT  -I A:c -O A:c′       # implemented as Toffoli with ancilla  
AND  -I A:c B:c -O T:c′   # CCNOT(A,B,0p) + CNOT(0p→T)  
NAND -I A:c B:c -O T:c′   # CCNOT(A,B,1p) + CNOT(1p→T)  
OR   -I A:c B:c -O T:c′   # De Morgan construction + CNOT  
XOR  -I A:c B:c -O T:c′   # two CCNOTs into a temporary ancilla  
```

## Agent Instructions

To help Codex or other AI agents generate, complete, and refactor QPU `.txt` protocols correctly, follow these guidelines:

1. **AST Mapping**

   * Match each textual command to its corresponding AST node class name (e.g., `H` → `GateASTNode`, `JOIN` → `JoinASTNode`).
   * Ensure the correct number of `-I` inputs and single `-O` output tokens, as enforced by each node’s constructor.

2. **Token Formatting**

   * Always use `q:cycle` format for cycle-annotated tokens; omit `:cycle` only when the output cycle defaults to the current cycle.
   * Numeric tokens (pure digits) denote physical qubit indices; non-numeric strings are treated as custom registers.

3. **Flag Ordering**

   * Commands requiring inputs and outputs must list `-I` before `-O`.
   * Do not interleave parameters or flags: e.g.

     ```bash
     CNOT -I 0:2 -O 1:2
     ```

4. **Parameter Files & Subprocesses**

   * For reusable protocols, begin with a `PARAMS:` line defining named parameters.
   * Use `COMPILEPROCESS --NAME name file.txt -I p1 p2 …` to compile with parameter substitution; follow with `CALL name -I v1 v2 …`.

5. **Comments & Continuations**

   * Use `#` for end-of-line comments; block comments survive without AST interference (`/* … */`).
   * Use trailing backslashes (`\`) to continue long logical lines.

6. **State Management**

   * Use `SET` to initialize or override memory entries before gate application.
   * After multi-qubit operations, use `JOIN`/`SPLIT` to reshape registers.
   * Issue `MEASURE` only after desired operations to collapse and record states.

7. **Error Handling**

   * Avoid undefined registers: always `CREATETOKEN` before use if the register name is non-numeric.
   * Respect each node’s input count: gates expect exactly the number of inputs defined in `gate_handlers`.

8. **Best Practices**

   * Keep linear, cycle-by-cycle logic simple; prefer explicit cycle annotations for clarity.
   * Group related operations into subprocess files, compiled and invoked via `COMPILEPROCESS`/`CALL`.
   * Use derived gates (`AND`, `OR`, etc.) sparingly; inline primitive gate sequences can be clearer.

By adhering to these conventions, AI agents can reliably author syntactically correct, semantically meaningful QPU instruction files that parse cleanly through `qpu/ast.py` and execute on the Python QPU simulator.
