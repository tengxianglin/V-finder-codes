#!/usr/bin/env python3
"""
example_transpose_minimal_anticommuting_time.py

Example script to demonstrate the speed of find_minimal_anticommuting_set in mode='transpose'.
- Randomly generate a Pauli term V
- Use generate_transpose_items to create H_name_list based on inverted commute conditions of 'conj':
  * Items with an odd number of Y must anticommute with V
  * Items with an even number of Y must commute with V
- Time the execution of find_minimal_anticommuting_set(H_name_list, mode='transpose')
- Print the initial V, len(H_name_list), elapsed time, resulting V, and consistency check
"""
import time
from hamiltonian_utils import (
    generate_random_pauli,
    find_minimal_anticommuting_set,
    generate_qubit_ops,
    is_commute,
)


def generate_1slot_paulis_transpose(
    base_pauli: str, num_qubits: int, target_count: int
):
    """
    Generate Pauli terms according to 'transpose' logic:
      - Odd Y-count terms must anticommute with base_pauli
      - Even Y-count terms must commute with base_pauli
    """
    valid_items = []
    while len(valid_items) < target_count:
        candidates = generate_qubit_ops(num_qubits, target_count)
        for term in candidates:
            # count Y operations in term
            y_count = sum(
                bool(op.strip().upper().startswith("Y")) for op in term.split(",")
            )
            should_anticommute = y_count % 2 == 1
            if (
                (should_anticommute and not is_commute(base_pauli, term))
                or (not should_anticommute and is_commute(base_pauli, term))
            ) and term not in valid_items:
                valid_items.append(term)
                if len(valid_items) >= target_count:
                    break
    return valid_items


def main():
    # Settings
    num_qubits = 10
    target_count = 10**5  # Number of items to generate
    mode = "transpose"

    # Randomly generate a num_qubits-qubit Pauli term V
    V = generate_random_pauli(num_qubits)
    print(f"Initial randomly generated {num_qubits}-qubit Pauli term V: {V}")

    # Generate H_name_list using 'transpose' logic
    print(f"Generating {target_count} items with transpose conditions based on V...")
    H_name_list = generate_1slot_paulis_transpose(V, num_qubits, target_count)
    print(f"H_name_list generated, length: {len(H_name_list)}")

    # Time find_minimal_anticommuting_set in 'transpose' mode
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
