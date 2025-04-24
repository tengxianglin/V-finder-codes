# Usage Tutorial

This tutorial guides you through encoding a Pauli operator V, generating a test set S, solving for V, and verifying results.

## 1. Encoding a Known V

- A single-qubit Pauli is mapped:
  - I → 00
  - X → 01
  - Z → 10
  - Y → 11
- An N-qubit Pauli string V is represented as a 2N-bit vector:
  - x[i]=1 if V has X or Y on qubit i
  - z[i]=1 if V has Z or Y on qubit i
- Example:
  ```python
  V = "X0,Z2,Y3"
  # x = [1,0,0,1]; z = [0,0,1,1]
  ```

## 2. Generating Test Set S

1. Choose N (number of qubits) and M (size of set).
2. Randomly sample Pauli terms that anticommute or satisfy mode-specific relation with V:
   - Dagger mode (`mode='dagger'`): pick terms P such that {P, V}=0
   - Conjugate mode (`mode='conj'`): odd Y-count terms commute, even Y-count anticommute
   - Transpose mode (`mode='transpose'`): odd Y-count terms anticommute, even Y-count commute
3. Use utility functions in `hamiltonian_utils.py`:
   ```python
   from hamiltonian_utils import generate_random_pauli, generate_qubit_ops, is_commute
   # Example for dagger:
   def gen_dagger(V, N, M):
       S = []
       while len(S) < M:
           candidate = generate_qubit_ops(N, M)
           for term in candidate:
               if not is_commute(V, term):
                   S.append(term)
                   if len(S) == M:
                       break
       return S
   ```

## 3. Solving for V

- Call the solver API:
  ```python
  from hamiltonian_utils import find_pauli_mapping_operators
  result = find_pauli_mapping_operators(pauli_set=S, mode='dagger')
  # result is a set of Pauli strings encoding V
  computed_V = next(iter(result))
  ```

## 4. Checking Equality up to Global Phase

- Two Pauli strings `V1` and `V2` match if they list the same terms (order-insensitive).
- Example:
  ```python
  set(V1.split(",")) == set(V2.split(","))
  ```

## 5. Measuring Runtime

- Use Python’s `time` module:
  ```python
  import time
  start = time.time()
  _ = find_pauli_mapping_operators(pauli_set=S, mode='dagger')
  print("Elapsed:", time.time() - start)
  ```
- Or use `pytest-benchmark` and `pytest-timeout` for automated benchmarks.

## 6. Complete Example

```python
from hamiltonian_utils import generate_random_pauli, generate_qubit_ops, is_commute, find_pauli_mapping_operators
import time

N, M = 4, 1000
V = generate_random_pauli(N)
S = gen_dagger(V, N, M)
start = time.time()
result = find_pauli_mapping_operators(pauli_set=S, mode='dagger')
computed_V = next(iter(result))
print("Computed V:", computed_V)
print("Match:", set(computed_V.split(",")) == set(V.split(",")))
print("Time:", time.time() - start)
```
