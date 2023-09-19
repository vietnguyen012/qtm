import qiskit
import random
import qtm.evolution
import qtm.utilities 
import qtm.evolution.ecircuit

def bitflip_mutate(individual: qtm.evolution.ecircuit.ECircuit, pool, is_truncate=True):
    point = random.random()
    qc1, qc2 = qtm.evolution.utils.divide_circuit(individual.qc, point)
    qc1.barrier()
    qc21, qc22 = qtm.evolution.utils.divide_circuit_by_depth(qc2, 1)
    genome = qtm.random_circuit.generate_with_pool(individual.qc.num_qubits, 1, pool)
    new_qc = qtm.utilities.compose_circuit([qc1, genome, qc22])
    if is_truncate:
        if new_qc.depth() > individual.qc.depth():
            new_qc, _ = qtm.evolution.utils.divide_circuit_by_depth(
                new_qc, individual.qc.depth())
    individual.qc = new_qc
    return individual
