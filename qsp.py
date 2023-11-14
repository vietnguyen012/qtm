import pickle
import qtm.metric
import qiskit
import numpy as np
import types, typing
import os

class QuantumStatePreparation:
    def __init__(self, file_name: str):
        """Load QSP from .qspobj file

        Args:
            file_name (str): Path to file

        """
        file = open(file_name, 'rb')
        data = pickle.load(file)
        self.__init__(
            data.u,
            data.vdagger,
            data.thetas,
            data.ansatz
        )
        file.close()
        return

    def __init__(self, u: typing.Union[qiskit.QuantumCircuit, str], 
                 vdagger: qiskit.QuantumCircuit = None, 
                 thetas: np.ndarray = None, 
                 ansatz: types.FunctionType = None):
        """There are four key atttributes for QSP problem: u, vdagger, parameters of u and name of u.

        Args:
            u (qiskit.QuantumCircuit): Ansatz
            vdagger (qiskit.QuantumCircuit): Prepare state
            thetas (np.ndarray): Optimized parameters
            ansatz (types.FunctionType): Name of u

        Returns:
            QuantumStatePreparation: completed object
        """

        if isinstance(u, str):
            file = open(u, 'rb')
            data = pickle.load(file)
            self.u = data.u
            self.vdagger = data.vdagger
            self.thetas = data.thetas
            self.ansatz = data.ansatz
        else:
            self.u = u
            self.vdagger = vdagger
            self.thetas = thetas
            self.ansatz = ansatz

        traces, fidelities = qtm.metric.calculate_compilation_metrics(
            self.u, self.vdagger, np.expand_dims(self.thetas, axis=0))
        self.trace = traces[0]
        self.fidelity = fidelities[0]
        self.num_qubits = self.u.num_qubits
        self.num_params = len(self.u.parameters)
        self.num_layers = int(
            self.num_params/len(self.ansatz(self.num_qubits, 1).parameters))
        self.qc = self.u.assign_parameters(self.thetas)
        return

    # def __init__(self, compiler, ansatz: types.FunctionType):
    #     """Load QSP from its parent (Compiler)

    #     Args:
    #         compiler (qtm.qcompilation.QuantumCompilation): Parent/Superset of QSP
    #         ansatz (types.FunctionType): Name of u

    #     """
    #     return self.__init__(self, compiler.u, compiler.vdagger, compiler.thetass[-1], ansatz)

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

    @staticmethod
    def find_satisfying_qspobj(state: str, num_qubits: int, error_rate: float, database_path: str):
        best_qspobj = None
        files = [f for f in os.listdir(database_path) if os.path.isfile(os.path.join(database_path, f))]
        files = [f for f in files if f.split('_')[0] == state and int(f.split('_')[2]) == num_qubits]
        print(files)
        for i in range(0, len(files)):
            path = database_path + files[i]
            qspobj = QuantumStatePreparation(path)
            if qspobj.fidelity > 1 - error_rate:
                if best_qspobj is None:
                    best_qspobj = qspobj
                if qspobj.u.depth() < best_qspobj.u.depth():
                    best_qspobj = qspobj
                elif qspobj.u.depth() == best_qspobj.u.depth():
                    if qspobj.num_params < best_qspobj.num_params:
                        best_qspobj = qspobj
            
                print(best_qspobj.fidelity,best_qspobj.u.depth(),best_qspobj.num_params)             
            
        if best_qspobj is None:
            print(f"Can not find the existing ansatz which can prepare state {state} {num_qubits} qubits >= {1 - error_rate} fidelity")
            return
        else:
            layer_verb = 'layer' if best_qspobj.num_layers == 1 else 'layers' 
            print(f"Found {best_qspobj.ansatz.__name__} {best_qspobj.num_layers} {layer_verb} which can prepare state {state} {num_qubits} qubits >= {1 - error_rate} fidelity ({best_qspobj.fidelity})")
            return best_qspobj


