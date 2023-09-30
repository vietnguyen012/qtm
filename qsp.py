import pickle
import qtm.utilities
import qiskit
import numpy as np
import types

class QuantumStatePreparation:
    def __init__(self, u: qiskit.QuantumCircuit, vdagger: qiskit.QuantumCircuit, thetas: np.ndarray, ansatz: types.FunctionType):
        """There are four key atttributes for QSP problem: u, vdagger, parameters of u and name of u.

        Args:
            u (qiskit.QuantumCircuit): Ansatz
            vdagger (qiskit.QuantumCircuit): Prepare state
            thetas (np.ndarray): Optimized parameters
            ansatz (types.FunctionType): Name of u

        Returns:
            QuantumStatePreparation: completed object
        """
        self.u = u
        self.vdagger = vdagger
        self.thetas = thetas
        self.ansatz = ansatz
        
        self.trace, self.fidelity = qtm.utilities.calculate_QSP_metric(
            u, vdagger, thetas)
        self.num_qubits = u.num_qubits
        self.num_params = len(self.u.parameters)
        self.num_layers = int(
            self.num_params/len(self.ansatz(self.num_qubits, 1).parameters))
        self.qc = self.u.bind_parameters(self.thetas)
        return self

    @classmethod
    def load_from_compiler(self, compiler, ansatz: types.FunctionType):
        """Load QSP from its parent (Compiler)

        Args:
            compiler (qtm.qcompilation.QuantumCompilation): Parent/Superset of QSP
            ansatz (types.FunctionType): Name of u

        """
        return self.__init__(self, compiler.u, compiler.vdagger, compiler.thetass[-1], ansatz)

    @classmethod
    def load_from_file(self, file_name: str):
        """Load QSP from .qspobj file

        Args:
            file_name (str): Path to file

        """
        file = open(file_name, 'rb')
        data = pickle.load(file)
        return self.__init__(self, data.u, data.vdagger, data.thetas, data.ansatz)
    @classmethod
    def save(self, state: str, file_name: str):
        """Save QSP to .qspobj file with a given path

        Args:
            state (str): Name of vdagger
            file_name (str): Saved path
        """
        file = open(
            f'{file_name}/{state}_{self.ansatz.__name__}_{self.num_qubits}_{self.num_layers}.qspobj', 'wb')
        pickle.dump(self, file)
        file.close()
        return
