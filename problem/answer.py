import sys
from typing import Any

######################
from qiskit_algorithms.optimizers import SLSQP, P_BFGS, NELDER_MEAD, SPSA
import qiskit
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
from tqdm import tqdm
import marshal
import csv

from openfermion.transforms import jordan_wigner

from quri_parts.algo.ansatz import SymmetryPreservingReal
from qiskit_algorithms.optimizers import SLSQP, P_BFGS, NELDER_MEAD, SPSA
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit, UnboundParametricQuantumCircuit
from quri_parts.core.estimator.gradient import parameter_shift_gradient_estimates
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement, CachedMeasurementFactory
from quri_parts.core.sampling.shots_allocator import (
    create_equipartition_shots_allocator,
)
from quri_parts.core.state import ParametricCircuitQuantumState, ComputationalBasisState, GeneralCircuitQuantumState

from quri_parts.openfermion.operator import operator_from_openfermion_op
#########################

sys.path.append("../")
from utils.challenge_2024 import ChallengeSampling, problem_hamiltonian

challenge_sampling = ChallengeSampling()

"""
####################################
add codes here
####################################
"""

total_shots = 10**3
shots_allocator = create_equipartition_shots_allocator()
measurement_factory = bitwise_commuting_pauli_measurement

estimator = challenge_sampling.create_concurrent_parametric_sampling_estimator(
    n_shots=total_shots,
    measurement_factory=measurement_factory,
    shots_allocator=shots_allocator,
)

def su4(qubits,circuit):
    

    circuit.add_ParametricRY_gate(qubits[0])
    circuit.add_ParametricRX_gate(qubits[0])
    circuit.add_ParametricRY_gate(qubits[0])
    
    circuit.add_ParametricRY_gate(qubits[1])
    circuit.add_ParametricRX_gate(qubits[1])
    circuit.add_ParametricRY_gate(qubits[1])
    
    circuit.add_CNOT_gate(qubits[0],qubits[1])
    
    circuit.add_ParametricRZ_gate(qubits[0])
    circuit.add_ParametricRY_gate(qubits[1])

    circuit.add_CNOT_gate(qubits[1],qubits[0])
    
    circuit.add_ParametricRY_gate(qubits[1])
    
    circuit.add_CNOT_gate(qubits[0],qubits[1])
    
    circuit.add_ParametricRY_gate(qubits[0])
    circuit.add_ParametricRX_gate(qubits[0])
    circuit.add_ParametricRY_gate(qubits[0])
    
    circuit.add_ParametricRY_gate(qubits[1])
    circuit.add_ParametricRX_gate(qubits[1])
    circuit.add_ParametricRY_gate(qubits[1])

def apply_su4_in_layers(depth, nqubits, fix_pars_per_layer=False):
    """Apply a (callable) operation in layers.

    Args:
        theta (tensor_like): The arguments passed to the operations. The expected shape
            is ``(depth, k, num_params_op)``, where ``k`` is determined by ``fix_pars_per_layer``
            and ``num_params_op`` is the number of paramters each operation takes.
        Op (callable): The operation to apply
        depth (int): The number of layers to apply
        fix_pars_per_layer (bool): Whether or not all operations applied in parallel share the
            same parameters. If ``True``, the dimension ``k`` in the shape of ``theta`` is set to
            two, otherwise it is set to ``nqubits``.

    """
    circuit = UnboundParametricQuantumCircuit(nqubits)
    #circuit = LinearMappedUnboundParametricQuantumCircuit(n_qubits)
    qubits = list(range(nqubits))
    
    if fix_pars_per_layer:
        size = 2
    else:
        size = nqubits
    
    for d in range(depth):
        # Even-odd qubit pairs
        idx = 0
        for q in qubits[0::2]:
            #`su4(theta[(d*size + idx)*15:(d*size + idx +1)*15], [q, (q+1+d)%nqubits],circuit)
            su4([q, (q+1+d)%nqubits],circuit)
            if q == nqubits - 2:
                idx += 1
            elif not fix_pars_per_layer:
                idx += 1
            
        # Odd-even qubit pairs
        for q in qubits[1::2]:
            # su4(theta[(d*size + idx)*15:(d*size + idx +1)*15], [q, (q+1+d)%nqubits],circuit)
            su4([q, (q+1+d)%nqubits],circuit)
            if not fix_pars_per_layer:
                idx += 1
    return circuit

def vqe(
    num_qubits,
    observable,
    depth,
    optimizer,
    max_steps,
    init_param=None):

    # n = 4**num_SUgate_qubits - 1 = 15
    param_len = depth*num_qubits*15
    
    # Initialize Hartree-Fock state
    n_site = num_qubits // 2
    hf_gates = ComputationalBasisState(num_qubits, bits=2**n_site - 1).circuit.gates
    circuit = UnboundParametricQuantumCircuit(num_qubits).combine(
            hf_gates
        )
    circuit.extend(apply_su4_in_layers(depth, num_qubits, fix_pars_per_layer=False))
    circuit_state = ParametricCircuitQuantumState(num_qubits, circuit)
    
    def cost_function(params):      
        estimated_value = estimator(observable,circuit_state,[params])[0].value.real
        return estimated_value
    
    def grad_function(params):
        grad = parameter_shift_gradient_estimates(
            observable, circuit_state, params, estimator
        )
        return np.asarray([i.real for i in grad.values])

    # Create initial parameters
    if init_param is None:
        init_params = np.array([0.1]*param_len)
    else:
        init_params = init_param

    Optimizer = optimizer(maxiter=max_steps)
    # Optimizer = optimizer(maxfun=max_steps)
    opt_result = Optimizer.minimize(fun=cost_function, x0=init_params, jac=grad_function)

    return opt_result

def separat_vqe(
    num_qubits,
    observable,
    depth,
    optimizer,
    separat_steps,
    init_param=None):
    
   
    if init_param is None:
        result = vqe(num_qubits,
                     observable,
                     depth,
                     optimizer=optimizer[0],
                     max_steps=separat_steps[0],
                     init_param=None)
    else:
        result = vqe(num_qubits,
                     observable,
                     depth,
                     optimizer=optimizer[0],
                     max_steps=separat_steps[0],
                     init_param=init_param)
    print(result.fun)
    #statewriter.writerow(result.x)
    #print(result.x)
    
    
    tot_step = separat_steps[0]
    for i in range(len(separat_steps)-1):
        result = vqe(num_qubits,
                     observable,
                     depth,
                     optimizer=optimizer[i+1],
                     max_steps=separat_steps[i+1],
                     init_param=result.x)
        tot_step += separat_steps[i+1]
        print(result.fun)
        #statewriter.writerow(result.x)
        #print(result.x)
        print("completed steps:",tot_step)
        
    return result.fun
"""
####################################
add codes here
####################################
"""



class RunAlgorithm:
    def __init__(self) -> None:
        challenge_sampling.reset()

    def result_for_evaluation(self, seed: int, hamiltonian_directory: str) -> tuple[Any, float]:
        energy_final = self.get_result(seed, hamiltonian_directory)
        total_shots = challenge_sampling.total_shots
        return energy_final, total_shots

    def get_result(self, seed: int, hamiltonian_directory: str) -> float:
        """
            param seed: the last letter in the Hamiltonian data file, taking one of the values 0,1,2,3,4
            param hamiltonian_directory: directory where hamiltonian data file exists
            return: calculated energy.
        """
        n_qubits = 28
        ham = problem_hamiltonian(n_qubits, seed, hamiltonian_directory)
        """
        ####################################
        add codes here
        ####################################
        """
        n_site = n_qubits // 2
        jw_hamiltonian = jordan_wigner(ham)
        hamiltonian = operator_from_openfermion_op(jw_hamiltonian)
       
        
        result = vqe(
            n_qubits,
            observable=hamiltonian,
            depth=2,
            optimizer=SLSQP,
            max_steps=100,
            init_param=None)
        """
        ####################################
        add codes here
        ####################################
        """

        return result.fun


if __name__ == "__main__":
    run_algorithm = RunAlgorithm()
    print(run_algorithm.get_result(seed=0, hamiltonian_directory="../hamiltonian"))
