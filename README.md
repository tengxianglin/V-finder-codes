# Find Pauli $V$ for $U^\dagger$, $U^T$, $U^\ast$

One-line description: Find the Pauli operator $V$ that maps a given Pauli operator U to its inverse, transpose, or conjugate via:

$$VUV = U^\dagger$$

$$VUV = U^T$$

$$VUV = U^\ast$$

## Background

The Pauli group on $N$-qubits consists of tensor products of single-qubit operators $\{I, X, Y, Z\}$. Each Pauli can be mapped to a 2-bit representation:
- $I \to 00$
- $X \to 01$
- $Z \to 10$
- $Y \to 11$

An $N$-qubit Pauli string is encoded as a $2N$-bit vector $(x \mid z)$, where $x$ and $z$ are $N$-bit binary vectors indicating the presence of $X$ and $Z$ on each qubit. $Y$ is encoded as $x_i = 1$ and $z_i = 1$.

A Pauli operator $V$ satisfies anti-commutation relations that implement conjugation, transposition, or inversion.  This solver uses Gaussian elimination over $GF(2)$ to find $V$ given a set of Pauli terms $S$ that anticommute under the chosen mapping.

## Installation & Dependencies

Install QuAIRKit and its dependencies by following the instructions at https://github.com/QuAIR/QuAIRKit.

## Usage Examples

### Command-line Interface (CLI)

To run the test harness for each mode on $N=10$, $M=10^5$:

```bash
python calculate_V_for_dagger.py
python calculate_V_for_transpose.py
python calculate_V_for_conj.py
```

### API

```python
from hamiltonian_utils import find_pauli_mapping_operators

# H_name_list: list of Pauli strings that anticommute under chosen mode
V = find_pauli_mapping_operators(H_name_list, mode='dagger')
```

## Test Harness

1. Randomly pick an N-qubit Pauli V.
2. Generate $M$ random Pauli terms $P_j$ that anticommute with $V$ (mode-specific).
3. Call:
   - `find_pauli_mapping_operators(H_name_list, mode='dagger')`
   - `find_pauli_mapping_operators(H_name_list, mode='transpose')`
   - `find_pauli_mapping_operators(H_name_list, mode='conj')`
4. Verify returned $V$ matches the original up to a global phase.
5. Measure wall-clock runtime (target: $N=10$, $M=10^5$).

## API Reference

### find_pauli_mapping_operators(pauli_set: Set[str], mode: str) → Set[str]
- pauli_set: Set of Pauli strings (e.g., "X0,Z1,Y2").
- mode: one of 'dagger', 'transpose', 'conj'.
- Returns: Set of Pauli strings representing $V$ in minimal anticommuting basis.

## Running the Large-Scale Test

```bash
# Generate 10^5 items and solve for N=10
python calculate_V_for_dagger.py
```

Sample Output:
```
Initial V: X0,Y3,Z7,...
Generating 100000 items...
find_pauli_mapping_operators elapsed time: 0.85 seconds
First element of result set V: X0,Y3,Z7,...
Correct: True
```
