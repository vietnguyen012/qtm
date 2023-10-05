import qtm.utilities
from qtm.evolution import utils
from .ecircuit import ECircuit
    
def onepoint_crossover(circuit1, circuit2, percent = None):
    if percent is None:
        percent = 1 - circuit1.fitness / (circuit1.fitness+ circuit2.fitness)
    sub11, sub12 = utils.divide_circuit_by_depth(circuit1.qc, int(percent*circuit2.qc.depth()))
    sub21, sub22 = utils.divide_circuit_by_depth(circuit2.qc, int((1 - percent) * circuit2.qc.depth()))
    new_qc1 = ECircuit(qtm.utilities.compose_circuit([sub11, sub22]), circuit1.fitness_func)
    new_qc2 = ECircuit(qtm.utilities.compose_circuit([sub21, sub12]), circuit1.fitness_func)
    return new_qc1, new_qc2

    