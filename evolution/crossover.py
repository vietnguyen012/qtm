import qtm.utilities
from qtm.evolution import utils
from .ecircuit import ECircuit
    
def onepoint_crossover(qc1, qc2, percent = 0.5):
    percent_sub1 = 1 - qc1.fitness / (qc1.fitness+ qc2.fitness)
    sub11, sub12 = utils.divide_circuit(qc1.qc, percent_sub1)
    sub21, sub22 = utils.divide_circuit(qc2.qc, 1 - percent_sub1)
    new_qc1 = ECircuit(qtm.utilities.compose_circuit([sub11, sub22]), qc1.fitness_func)
    new_qc2 = ECircuit(qtm.utilities.compose_circuit([sub21, sub12]), qc1.fitness_func)
    return new_qc1, new_qc2

