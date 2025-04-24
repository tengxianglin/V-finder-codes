#!/usr/bin/env python3
"""
example_dagger_minimal_anticommuting_time.py

Example script to demonstrate the speed of find_pauli_mapping_operators in mode='dagger'.
- Randomly generate a Pauli term V
- Use generate_anticommuting_items to create H_name_list
- Time the execution of find_pauli_mapping_operators(H_name_list, mode='dagger')
- Print the initial V, len(H_name_list), elapsed time, resulting V, and consistency check
"""
import time
from typing import Set
from hamiltonian_utils import (
    find_pauli_mapping_operators,
    generate_qubit_ops,
    generate_random_pauli,
    is_commute,
)

import torch

# torch.set_default_device("cuda")  # Set default device to GPU if available


def generate_1slot_paulis_dagger(base_pauli: str, num_qubits: int, target_count: int):
    """
    Generate a list of Pauli terms that anticommute with a given base term.

    Args:
        base_pauli (str): Reference Pauli string.
        num_qubits (int): Number of qubits.
        target_count (int): Desired number of items.

    Returns:
        List[str]: Terms anticommute with base_pauli.
    """
    valid_items: Set[str] = set()
    while len(valid_items) < target_count:
        candidates = generate_qubit_ops(num_qubits, target_count)
        for term in candidates:
            if term in valid_items:
                continue
            if not is_commute(base_pauli, term):
                valid_items.add(term)
                if len(valid_items) >= target_count:
                    break
    return valid_items


def main():
    # Settings
    num_qubits = 10
    target_count = 10**5  # Number of items to generate
    mode = "dagger"

    # Randomly generate a num_qubits-qubit Pauli term V
    V = generate_random_pauli(num_qubits)
    print(f"Initial randomly generated {num_qubits}-qubit Pauli term V: {V}")

    # Generate pauli_set of anticommuting items
    print(f"Generating {target_count} anticommuting items based on V...")
    pauli_set = generate_1slot_paulis_dagger(V, num_qubits, target_count)
    print(f"pauli_set generated, size: {len(pauli_set)}")

    # Time find_pauli_mapping_operators
    print(f"Starting timer and running find_pauli_mapping_operators(mode='{mode}')...")
    start_time = time.time()
    result_set = find_pauli_mapping_operators(pauli_set=pauli_set, mode=mode)
    elapsed = time.time() - start_time
    print(f"find_pauli_mapping_operators elapsed time: {elapsed:.6f} seconds")

    # Extract and print results, ensure exactly one element matching initial V
    if len(result_set) != 1:
        raise RuntimeError(
            f"Expected single result V, got {len(result_set)}: {result_set}"
        )
    found_V = next(iter(result_set))
    print(f"Computed V from minimal anticommuting set: {found_V}")
    print(f"Matches initial V: {found_V == V}")


if __name__ == "__main__":
    main()
