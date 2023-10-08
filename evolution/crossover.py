import qtm.utilities
from qtm.evolution import utils
from .ecircuit import ECircuit
import qiskit


def onepoint_crossover(circuit1: ECircuit, circuit2: ECircuit, percent: float = None, is_truncate=True):
    """Cross over between two circuits and create 2 offsprings

    Args:
        circuit1 (qtm.evolution.ecircuit.ECircuit): Father
        circuit2 (qtm.evolution.ecircuit.ECircuit): Mother
        percent (float, optional): Percent of father's genome in offspring 1. Defaults to None.

    """
    # If percent is not produced, dividing base on how strong of father's fitness.
    standard_depth = circuit1.qc.depth()
    standard_fitness_func = circuit1.fitness_func
    if percent is None:
        percent = 1 - circuit1.fitness / (circuit1.fitness + circuit2.fitness)
    sub11, sub12 = qtm.utilities.divide_circuit_by_depth(
        circuit1.qc, int(percent*standard_depth))
    sub21, sub22 = qtm.utilities.divide_circuit_by_depth(
        circuit2.qc, int((1 - percent) * standard_depth))
    combined_qc1 = qtm.utilities.compose_circuit([sub11, sub22])
    combined_qc2 = qtm.utilities.compose_circuit([sub21, sub12])
    if is_truncate:
        combined_qc1 = qtm.utilities.truncate_circuit(
            combined_qc1, standard_depth)
        combined_qc2 = qtm.utilities.truncate_circuit(
            combined_qc2, standard_depth)
    new_qc1 = ECircuit(combined_qc1, standard_fitness_func)
    new_qc2 = ECircuit(combined_qc2, standard_fitness_func)
    return new_qc1, new_qc2
