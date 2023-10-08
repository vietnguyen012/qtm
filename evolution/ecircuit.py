import types
import qiskit

class ECircuit():
    def __init__(self, qc: qiskit.QuantumCircuit, fitness_func: types.FunctionType) -> None:
        """Enhanced qiskit circuit with fitness properties

        Args:
            qc (qiskit.QuantumCircuit)
            fitness_func (types.FunctionType)
        """
        self.qc = qc
        self.fitness_func = fitness_func
        self.fitness = 0
        self.compile()
        return
    def compile(self):
        """Run fitness function to compute fitness value
        """
        self.fitness = self.fitness_func(self.qc)
        return
