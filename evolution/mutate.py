import qiskit
import random
import qtm.evolution
import qtm.utilities 
import qtm.evolution.ecircuit
import qtm.constant

def bitflip_mutate(circuit: qtm.evolution.ecircuit.ECircuit, pool, is_truncate=True):
    point = random.random()
    qc1, qc2 = qtm.evolution.utils.divide_circuit(circuit.qc, point)
    qc21, qc22 = qtm.evolution.utils.divide_circuit_by_depth(qc2, 1)
    genome = qtm.random_circuit.generate_with_pool(circuit.qc.num_qubits, 1, pool)
    new_qc = qtm.utilities.compose_circuit([qc1, genome, qc22])
    if is_truncate:
        if new_qc.depth() > circuit.qc.depth():
            new_qc, _ = qtm.evolution.utils.divide_circuit_by_depth(
                new_qc, circuit.qc.depth())
    circuit.qc = new_qc
    return circuit
