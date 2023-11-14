import types, typing
import qiskit
from ..constant import NORMAL_MODE, PREDICTOR_MODE
class ECircuit():
    def __init__(self, qc: qiskit.QuantumCircuit, fitness_func: types.FunctionType, mode = NORMAL_MODE) -> None:
        """Enhanced qiskit circuit with fitness properties

        Args:
            qc (qiskit.QuantumCircuit)
            fitness_func (types.FunctionType)
        """
        self.qc = qc
        self.fitness_func = fitness_func
        self.fitness = 0
        self.true_fitness = 0
        self.strength_point = 0
        self.mode = mode
        return
    def compile(self):
        """Run fitness function to compute fitness value
        """
        self.fitness = self.fitness_func(self.qc)
        return
    def true_compile(self):
        self.true_fitness = self.fitness_func(self.qc, num_steps = 100)
        return