# Quantum Algorithm Grand Challenge

# Table of Contents

1. [Overview of QAGC](#Overview)
2. [Problem Description](#Description)
3. [Methodology](#methods)
4. [Results](#Results)
5. [Conclusion](#Conclusion)


# Overview Of QAGC <a id="Overview"></a>
Quantum Algorithm Grand Challenge (QAGC) is a global online contest for students, researchers, and others who learn quantum computation and quantum chemistry around the world.

QAGC 2024 aims explore practical uses for NISQ devices, visualize bottlenecks in NISQ device utilization, and create a metric for benchmarking the NISQ algorithms. 

This year's challenge is to find the ground-state energy of a given model (Hamiltonian).

The score is calculated as the average accuracy computed from each absolute error over 10 runs of the algorithm, rounded to the nearest. The smaller the score, the higher the ranking can be achieved.

# Problem Description <a id="Description"></a>
The hamiltonian used is QAGC is the [Fermi-Hubbard Model](https://arxiv.org/abs/2402.11869). The problem statement is as follows:

Find the energy of the ground state of the one-dimensional orbital rotated Fermi-Hubbard model.

$$
    \tilde{H} = - t \sum_{i=0}^{2N-1}(\tilde c^\dagger_i \tilde c_{i+1} + \tilde c^\dagger_{i+1} \tilde c_i)  - \mu \sum_{i=0}^{2N-1}  \tilde c^\dagger_i \tilde c_i + U \sum_{i=0}^{N-1} \tilde c^\dagger_{2i} \tilde c_{2i} \tilde c^\dagger_{2i + 1} \tilde c_{2i + 1} 
$$

The value of each parameter is $N = 14,\ t=1, U=3,\ \mu=U/2 = 1.5$. 

Hamiltonians are provides for `num_qubits = [4, 8, 20, 28]`


# Methodology <a id="Methods"></a>

- Prepapring the ansatz
- Gradient

# Results <a id="Results"></a>
- Results with SU() initial gradient.
- Results with new code
- Results with optimizers (SLSQP, BFGS, Mender)
- Final result for 4, 8, 12, 20 qubits

# Conclusion <a id="Conclusion"></a>
