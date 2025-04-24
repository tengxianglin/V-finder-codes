#!/usr/bin/env python3
"""
example_dagger_minimal_anticommuting_time.py

Example script to demonstrate the speed of find_minimal_anticommuting_set in mode='dagger'.
- Randomly generate a Pauli term V
- Use generate_anticommuting_items to create H_name_list
- Time the execution of find_minimal_anticommuting_set(H_name_list, mode='dagger')
- Print the initial V, len(H_name_list), elapsed time, resulting V, and consistency check
"""
import time
from hamiltonian_utils import (
    find_minimal_anticommuting_set,
    generate_qubit_ops,
    generate_random_pauli,
    is_commute,
)


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
    valid_items = []
    while len(valid_items) < target_count:
        candidates = generate_qubit_ops(num_qubits, target_count)
        for term in candidates:
            if not is_commute(base_pauli, term) and term not in valid_items:
                valid_items.append(term)
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

    # Generate H_name_list
    print(f"Generating {target_count} anticommuting items based on V...")
    H_name_list = generate_1slot_paulis_dagger(V, num_qubits, target_count)
    print(f"H_name_list generated, length: {len(H_name_list)}")

    # Time find_minimal_anticommuting_set
    print(
        f"Starting timer and running find_minimal_anticommuting_set(mode='{mode}')..."
    )
    start_time = time.time()
    result_set = find_minimal_anticommuting_set(H_name_list=H_name_list, mode=mode)
    elapsed = time.time() - start_time
    print(f"find_minimal_anticommuting_set elapsed time: {elapsed:.6f} seconds")

    # Extract and print results
    found_V = result_set[0] if result_set else None
    print(f"First element of result set V: {found_V}")
    print(f"Does initial V match computed V? {found_V == V}")


if __name__ == "__main__":
    main()
