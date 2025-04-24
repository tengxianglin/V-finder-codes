#!/usr/bin/env python3
"""
example_transpose_minimal_anticommuting_time.py

Example script to demonstrate the speed of find_pauli_mapping_operators in mode='transpose'.
- Randomly generate a Pauli term V
- Use generate_transpose_items to create H_name_list based on inverted commute conditions of 'conj':
  * Items with an odd number of Y must anticommute with V
  * Items with an even number of Y must commute with V
- Time the execution of find_pauli_mapping_operators(H_name_list, mode='transpose')
- Print the initial V, len(H_name_list), elapsed time, resulting V, and consistency check
"""
import time
from typing import Set
from hamiltonian_utils import (
    generate_random_pauli,
    find_pauli_mapping_operators,
    generate_qubit_ops,
    is_commute,
)


# torch.set_default_device("cuda")  # Set default device to GPU if available


def generate_1slot_paulis_transpose(
    base_pauli: str, num_qubits: int, target_count: int
):
    """
    Generate Pauli terms according to 'transpose' logic:
      - Odd Y-count terms must anticommute with base_pauli
      - Even Y-count terms must commute with base_pauli
    """
    valid_items: Set[str] = set()
    while len(valid_items) < target_count:
        candidates = generate_qubit_ops(num_qubits, target_count)
        for term in candidates:
            if term in valid_items:
                continue
            # count Y operations in term
            y_count = sum(
                bool(op.strip().upper().startswith("Y")) for op in term.split(",")
            )
            should_anticommute = y_count % 2 == 1
            if (should_anticommute and not is_commute(base_pauli, term)) or (
                not should_anticommute and is_commute(base_pauli, term)
            ):
                valid_items.add(term)
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

    # Generate pauli_set using 'transpose' logic
    print(f"Generating {target_count} items with transpose conditions based on V...")
    pauli_set = generate_1slot_paulis_transpose(V, num_qubits, target_count)
    print(f"pauli_set generated, size: {len(pauli_set)}")

    # Time find_pauli_mapping_operators in 'transpose' mode
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
