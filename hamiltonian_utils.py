#!/usr/bin/env python3
"""
hamiltonian_utils.py

Utility module for finding a Pauli operator V that maps U to U†, Uᵀ, or U* by solving minimal anticommuting sets.

Pauli-to-binary mapping (single-qubit):
  I → 00
  X → 01
  Z → 10
  Y → 11  (X and Z both present)

An N-qubit Pauli string is encoded as a 2N-bit vector (x|z), where x[i]=1 if X or Y on qubit i, z[i]=1 if Z or Y on qubit i.

Gaussian elimination over GF(2) solves V by constructing a linear system A x = b, where rows correspond to input Paulis and b encodes the mode ('dagger','conj','transpose').
Complexity: O(M·N^2) for M terms and N qubits.

Exports:
  - is_commute: check Pauli commute relation
  - generate_all_paulis, generate_random_pauli, generate_qubit_ops
  - find_pauli_mapping_operators: solve for V via Gaussian elimination
"""
import itertools
import os
import pickle
import random

from typing import List, Set, Tuple

import torch


def is_commute(H_A: str, H_B: str) -> bool:
    items_A = H_A.split(",")
    items_B = H_B.split(",")

    dict_A = {}
    dict_B = {}

    for item in items_A:
        operator = item[0]
        position = int(item[1:])
        dict_A[position] = operator

    for item in items_B:
        operator = item[0]
        position = int(item[1:])
        dict_B[position] = operator

    anticommute_count = 0
    common_positions = set(dict_A.keys()) & set(dict_B.keys())

    for pos in common_positions:
        op_A = dict_A[pos]
        op_B = dict_B[pos]
        if op_A != op_B and sorted([op_A, op_B]) in [
            ["X", "Y"],
            ["Y", "Z"],
            ["X", "Z"],
        ]:
            anticommute_count += 1

    return anticommute_count % 2 != 1


def generate_all_paulis(num_qubits: int) -> List[str]:
    """Generate all possible Pauli strings with caching."""
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file path
    cache_file = os.path.join(cache_dir, f"paulis_{num_qubits}.pkl")

    # Try to load from cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            print(f"Cache file {cache_file} is corrupted, regenerating...")

    # Generate if not cached
    letters = ["I", "X", "Y", "Z"]
    all_possible = []
    for combo in itertools.product(letters, repeat=num_qubits):
        if any(letter != "I" for letter in combo):
            terms = [f"{letter}{i}" for i, letter in enumerate(combo) if letter != "I"]
            all_possible.append(",".join(terms))

    # Save to cache
    with open(cache_file, "wb") as f:
        pickle.dump(all_possible, f)

    return all_possible


def generate_random_pauli(num_qubits: int) -> str:
    """Generate a random Pauli string."""
    letters = ["I", "X", "Y", "Z"]
    term = []
    while not term:
        term = [
            f"{letter}{pos}"
            for pos, letter in enumerate(random.choices(letters, k=num_qubits))
            if letter != "I"
        ]
    return ",".join(term)


def generate_qubit_ops(num_qubits: int, num_item: int) -> List[str]:
    """
    Generate a list of Hamiltonian operators.

    Args:
        num_qubits: Number of qubits
        num_item: Number of operators to generate

    Returns:
        List of Pauli strings

    Raises:
        ValueError: If num_item is larger than the maximum possible number of operators
    """
    max_possible = 4**num_qubits - 1  # Subtract 1 to exclude identity

    if num_item > max_possible:
        raise ValueError(
            f"Cannot generate {num_item} operators: maximum possible is {max_possible}"
        )

    # If num_item is close to max_possible (>75%), generate all and sample
    if num_item * 1.618 > max_possible:
        all_paulis = generate_all_paulis(num_qubits)
        return random.sample(all_paulis, num_item)

    # Otherwise, generate randomly with checking for duplicates
    result = set()
    while len(result) < num_item:
        pauli = generate_random_pauli(num_qubits)
        result.add(pauli)

    return list(result)


# 1) Build augmented binary matrix (A|b) in one pass
def map_paulis_to_aug_matrix(pauli_set: Set[str], mode: str) -> Tuple[torch.Tensor, int]:
    """Map each Pauli in pauli_set to a single augmented row [x|z|b] over GF(2)."""
    # determine number of qubits
    num_qubits = (
        max(
            int(term[1:]) for name in pauli_set
            for term in name.split(",") if term and term[1:].isdigit()
        ) + 1
    )
    def pauli_to_bits(s: str):
        x = [0]*num_qubits; z = [0]*num_qubits
        for term in s.split(","):
            t = term.strip()
            if not t or t[0]=="I": continue
            op, idx = t[0], int(t[1:])
            if op=="X": x[idx]=1
            elif op=="Z": z[idx]=1
            elif op=="Y": x[idx]=1; z[idx]=1
        return x, z
    def get_b(x, z):
        if mode=="dagger":
            return 1
        same = sum(a&b for a,b in zip(x,z))
        if mode=="conj":
            return 1 if same%2==0 else 0
        if mode=="transpose":
            return 0 if same%2==0 else 1
        raise ValueError(f"Unknown mode: {mode}")
    rows = []
    for p in pauli_set:
        x, z = pauli_to_bits(p)
        rows.append(x + z + [get_b(x, z)])
    return torch.tensor(rows, dtype=torch.int32), num_qubits

# 2) Module‐level GF(2) Gaussian elimination
def gf2_solve(A_b: torch.Tensor) -> Tuple[bool, List[int]]:
    """In-place Gaussian elimination on A|b mod 2. Returns (True, solution_vector)."""
    A = A_b.clone()
    n_rows, m1 = A.shape
    n_vars = m1 - 1
    pivot_row = 0
    pivot_cols: List[int] = []
    for col in range(n_vars):
        if pivot_row>=n_rows: break
        sub = A[pivot_row:, col]
        nz = torch.nonzero(sub, as_tuple=False)
        if nz.numel()==0: continue
        r = pivot_row + nz[0].item()
        if r!=pivot_row:
            A[[pivot_row, r]] = A[[r, pivot_row]]
        mask = A[:, col].bool()
        mask[pivot_row] = False
        A[mask] ^= A[pivot_row]
        pivot_cols.append(col)
        pivot_row += 1
    # back‐substitute b‐values
    if pivot_cols:
        rows = torch.arange(len(pivot_cols), dtype=torch.long)
        cols = torch.tensor(pivot_cols, dtype=torch.long)
        bvals = A[rows, -1]
    else:
        cols = torch.empty(0, dtype=torch.long)
        bvals = torch.empty(0, dtype=torch.int32)
    x = torch.zeros(n_vars, dtype=torch.int32)
    x[cols] = bvals.to(torch.int32)
    return x.tolist()

# 3) Recursive solver: peel off one solution at a time
def recursive_gf2_solve(A_b: torch.Tensor) -> List[List[int]]:
    """Recursively solve A_b, collect each solution vector until all rows are removed."""
    solutions: List[List[int]] = []
    def _recurse(mat: torch.Tensor):
        if mat.size(0)==0:
            return
        sol = gf2_solve(mat)
        solutions.append(sol)
        num_vars = mat.size(1)-1
        nq = num_vars // 2
        z_sol = torch.tensor(sol[:nq], dtype=torch.int32)
        x_sol = torch.tensor(sol[nq:], dtype=torch.int32)
        x_mat, z_mat = mat[:, :nq], mat[:, nq:-1]
        b = mat[:, -1]
        dot = ((x_mat * z_sol) + (z_mat * x_sol)).sum(dim=1) % 2
        keep = dot != b
        if keep.any():
            _recurse(mat[keep])
    _recurse(A_b)
    return solutions

# 4) Convert solution bit‐vectors back to Pauli strings
def bits_matrix_to_pauli(solutions: List[List[int]], num_qubits: int) -> List[str]:
    """Map each [x|z] solution vector to its Pauli string representation."""
    paulis: List[str] = []
    for sol in solutions:
        z, x = sol[:num_qubits], sol[num_qubits:]
        terms = []
        for i, (qx, qz) in enumerate(zip(x, z)):
            if qx==1 and qz==0: terms.append(f"X{i}")
            elif qx==0 and qz==1: terms.append(f"Z{i}")
            elif qx==1 and qz==1: terms.append(f"Y{i}")
        paulis.append(",".join(terms) if terms else "I")
    return paulis


# Replace the old implementation with a single‐line pipeline
def find_pauli_mapping_operators(pauli_set: Set[str], mode: str) -> Set[str]:
    """Find minimal Pauli V’s by a one‐shot matrix pipeline."""
    A_b, num_qubits = map_paulis_to_aug_matrix(pauli_set, mode)
    sols = recursive_gf2_solve(A_b)
    return set(bits_matrix_to_pauli(sols, num_qubits))
