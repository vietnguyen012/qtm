import qiskit
import pickle


class QSP:
    def __init__(self, u: qiskit.QuantumCircuit = None,
                 vdagger: qiskit.QuantumCircuit = None, 
                 ansatz = None, 
                 num_layer = None, 
                 thetas = None, 
                 fidelity = None, 
                 trace = None):
        self.u = u
        self.vdagger = vdagger
        self.ansatz = ansatz
        self.num_layers = num_layer
        self.thetas = thetas
        self.fidelity = fidelity
        self.trace = trace
        
        self.num_qubits = u.num_qubits
        return

    def load(self, file_name):
        file = open(file_name, 'rb')
        data = pickle.load(file)
        self.__init__(data.u,
                      data.vdagger,
                      data.ansatz,
                      data.num_layer,
                      data.thetas,
                      data.fidelity,
                      data.trace)
        return

    def save(self, file_name):
        file = open(file_name, 'wb')
        pickle.dump(self, file)
        file.close()
        return
