#!/usr/bin/env python3
"""
anticommuting_utils.py

Utility module for generating Pauli terms under different commutation conditions
and finding minimal anticommuting sets.

Exports:
  - generate_anticommuting_items: generate terms that anticommute with a base term
  - generate_conj_items: generate terms for 'conj' mode
  - generate_transpose_items: generate terms for 'transpose' mode
  - find_minimal_anticommuting_set: wrapper to search minimal set (all modes)
"""
import itertools
import os
import pickle
import random

from typing import List, Tuple

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


def find_minimal_anticommuting_set(H_name_list: List[str], mode: str) -> List[str]:
    """
    Given a list of Pauli strings, find a minimal set of Pauli operators such that
    each operator in the input list anticommutes (or satisfies the specified mode) with at least one operator in the output set.

    Modes:
        - "dagger": b_value is always 1
        - "conj": b_value = 1 if parity of x*z is even else 0
        - "transpose": b_value = 0 if parity of x*z is even else 1 (opposite of conj)

    Args:
        H_name_list (List[str]): A list of Pauli strings (e.g., ["X0", "Y0"]).
        mode (str): "dagger", "conj", or "transpose".

    Returns:
        List[str]: A minimal set of Pauli strings that anticommute (or satisfy mode) with the input set.
    """

    def pauli_to_bits(pauli_str: str, num_qubits: int) -> Tuple[List[int], List[int]]:
        x_bits = [0] * num_qubits
        z_bits = [0] * num_qubits
        for term in pauli_str.split(","):
            term = term.strip()
            if not term or term[0] == "I":
                continue
            op, idx = term[0], int(term[1:])
            if op == "X":
                x_bits[idx] = 1
            elif op == "Z":
                z_bits[idx] = 1
            elif op == "Y":
                x_bits[idx] = 1
                z_bits[idx] = 1
        return x_bits, z_bits

    def bits_to_pauli(q_x: List[int], q_z: List[int]) -> str:
        terms = []
        for i, (qx, qz) in enumerate(zip(q_x, q_z)):
            if qx == 1 and qz == 0:
                terms.append(f"X{i}")
            elif qx == 0 and qz == 1:
                terms.append(f"Z{i}")
            elif qx == 1 and qz == 1:
                terms.append(f"Y{i}")
        return ",".join(terms) if terms else "I"

    def gf2_solve(A_b: torch.Tensor) -> Tuple[bool, List[int]]:
        A = A_b.clone()
        n_rows, m_plus1 = A.shape
        n_vars = m_plus1 - 1
        pivot_row = 0
        pivot_cols: List[int] = []
        for col in range(n_vars):
            if pivot_row >= n_rows:
                break
            sub = A[pivot_row:, col]
            nz = torch.nonzero(sub, as_tuple=False)
            if nz.numel() == 0:
                continue
            r = pivot_row + nz[0].item()
            if r != pivot_row:
                A[[pivot_row, r]] = A[[r, pivot_row]]
            mask = A[:, col].bool()
            mask[pivot_row] = False
            A[mask] ^= A[pivot_row]
            pivot_cols.append(col)
            pivot_row += 1
        if pivot_cols:
            row_idx = torch.arange(len(pivot_cols), dtype=torch.long)
            col_idx = torch.tensor(pivot_cols, dtype=torch.long)
            b_vals = A[row_idx, -1]
        else:
            row_idx = torch.empty(0, dtype=torch.long)
            col_idx = torch.empty(0, dtype=torch.long)
            b_vals = torch.empty(0, dtype=torch.int32)
        x = torch.zeros(n_vars, dtype=torch.int32)
        x[col_idx] = b_vals.to(torch.int32)
        return True, x.tolist()

    num_qubits = (
        max(
            int(term[1:])
            for name in H_name_list
            for term in name.split(",")
            if term[1:].isdigit()
        )
        + 1
    )

    pauli_bits = [pauli_to_bits(p, num_qubits) for p in H_name_list]

    def get_b_value(x, z, mode):
        if mode == "dagger":
            return 1
        elif mode == "conj":
            same_position_count = sum(a & b for a, b in zip(x, z))
            return 1 if same_position_count % 2 == 0 else 0
        elif mode == "transpose":
            same_position_count = sum(a & b for a, b in zip(x, z))
            return 0 if same_position_count % 2 == 0 else 1
        else:
            raise ValueError(f"Unknown mode: {mode}")

    covered = [False] * len(pauli_bits)
    Q = []

    while not all(covered):
        A_b = []
        for i, (x, z) in enumerate(pauli_bits):
            if not covered[i]:
                b_value = get_b_value(x, z, mode)
                A_b.append(x + z + [b_value])
        A_b = torch.tensor(A_b, dtype=torch.int32)
        ok, sol = gf2_solve(A_b)
        if ok:
            q_z = sol[:num_qubits]
            q_x = sol[num_qubits:]
        else:
            idx = covered.index(False)
            x0, z0 = pauli_bits[idx]
            q_x = [0] * num_qubits
            q_z = [0] * num_qubits
            if x0[0] == 1 and z0[0] == 0:
                q_z[0] = 1
            elif x0[0] == 0 and z0[0] == 1:
                q_x[0] = 1
            else:
                q_z[0] = 1
        q_str = bits_to_pauli(q_x, q_z)
        Q.append(q_str)
        for i, (x, z) in enumerate(pauli_bits):
            if not covered[i]:
                dot = (
                    sum(a & b for a, b in zip(x, q_z))
                    + sum(a & b for a, b in zip(z, q_x))
                ) % 2
                b_value = get_b_value(x, z, mode)
                if dot == b_value:
                    covered[i] = True
    return Q
