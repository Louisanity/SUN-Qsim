# Quantum Algorithm Grand Challenge 2024

## Table of Contents

1. [Overview of QAGC](#Overview)
2. [Problem Description](#Description)
3. [Methodology](#Methods)
4. [Results](#Results)
5. [Conclusion](#Conclusion)
6. [References](#References)

## Overview of QAGC <a id="Overview"></a>
The Quantum Algorithm Grand Challenge (QAGC) is a global online contest for students, researchers, and enthusiasts of quantum computation and quantum chemistry.

QAGC 2024 aims to explore practical uses for NISQ devices, visualize bottlenecks in NISQ device utilization, and create metrics for benchmarking NISQ algorithms.

This year's challenge is to find the ground-state energy of a given model (Hamiltonian).

Scores are calculated as the average accuracy computed from each absolute error over 10 runs of the algorithm, rounded to the nearest value. The smaller the score, the higher the ranking.

## Problem Description <a id="Description"></a>
The Hamiltonian used in QAGC is the [Fermi-Hubbard Model](https://arxiv.org/abs/2402.11869). The problem statement is as follows:

Find the energy of the ground state of the one-dimensional orbital-rotated Fermi-Hubbard model.


$$
    \tilde{H} = - t \sum_{i=0}^{2N-1}(\tilde c^\dagger_i \tilde c_{i+1} + \tilde c^\dagger_{i+1} \tilde c_i)  - \mu \sum_{i=0}^{2N-1}  \tilde c^\dagger_i \tilde c_i + U \sum_{i=0}^{N-1} \tilde c^\dagger_{2i} \tilde c_{2i} \tilde c^\dagger_{2i + 1} \tilde c_{2i + 1} 
$$

The values of each parameter are \(N = 14\), \(t = 1\), \(U = 3\), and \(\mu = \frac{U}{2} = 1.5\).

Hamiltonians are provided for `num_qubits = [4, 12, 20, 28]`.

## Methodology <a id="Methods"></a>
- Reading Hamiltonian
- Preparing the ansatz
- Initial State Preparation
- Gradient Calculation

## Results <a id="Results"></a>
- Metrics of resources used by Hamiltonians for qubits of different sizes:

| Hamiltonian Qubits | Memory Usage (%) | CPU Usage (%) | Time Taken (sec) |
|--------------------|------------------|---------------|------------------|
| 4                  | 2.5              | 14            | 2                |
| 12                 | 2.7              | 15.3          | 8                |
| 20                 | 2.9              | 16.2          | 72               |
| 28                 | 3.1              | 16            | 112              |

- Results with SU() initial gradient:
  Applying SU gates to qubits [0,1] in each layer.

- Results with new code:
  Changed the ordering: SU gate applied to [0,1] in the first layer, then [0,2] in the second layer.

### Using SLSQP for 12 qubits

**Depth 2**

| Depth | Steps | Value      | Error (%) |
|-------|-------|------------|-----------|
| 2     | 10    | -12.15445  | 9.3       |
| 2     | 20    | -12.4984   | 6.7       |
| 2     | 100   | -12.664    | 5.49      |
| 2     | 200   | -12.694    | 5.2       |

**Depth 3**

| Depth | Steps | Value   | Error (%) |
|-------|-------|---------|-----------|
| 3     | 100   | -12.685 | 5.34      |
| 3     | 200   | -12.69  | 5.29      |

### Using L_BFGS for 12 qubits

| Depth | Steps | Value      | Error (%) |
|-------|-------|------------|-----------|
| 1     | 10    | -12.498    | 6.73      |
| 1     | 100   | -12.49999  | 6.7       |
| 2     | 100   | -12.604    | 5.9       |
| 3     | 200   | -12.61     | 5.86      |

After trying various other optimizers like Nelder-Mead and P_BFGS, the configuration giving the best performance is as follows:

```
Optimizer: SLSQP
SU Ansatz: depth 2 steps 100
```
### Score Reference

| Qubits | Value  | Error (%) | Task Score |
|--------|--------|-----------|------------|
| 4      | -3.99  | 0.25      | 0.01       |
| 12     | -12.69 | 5.5       | 0.78       |
| 20     | -20.50 | 6.8       | 1.54       |
| 28     | -28.04 | 8.8       | 2.70       |

## Conclusion <a id="Conclusion"></a>

Our solution:

- Initializes state using Hartree-Fock Initialization
- Uses SU ansatz with depth 2
- Uses SLSQP Optimizer

## References <a id="References"></a>
- Wiersema, Roeland, et al. "Here comes the SU (N): multivariate quantum gates and gradients." Quantum 8 (2024): 1275.
- Kojima, Ryota, Masahiko Kamoshita, and Keita Kanno. "Orbital-rotated Fermi-Hubbard model as a benchmarking problem for quantum chemistry with the exact solution." arXiv preprint arXiv:2402.11869 (2024).
- 
## How to cite
If you used this solution and algorithm for your research, please cite:
```text
Under Preparation....
```
