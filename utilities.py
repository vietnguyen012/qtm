import typing
import numpy as np
import qiskit
import scipy
import qtm.constant
import tqdm
# Early stopping
class EarlyStopping:
    def __init__(self, patience=0, delta=0):
        """Class for early stopper

        Args:
            patience (int, optional): The number of unchanged loss values step. Defaults to 0.
            delta (int, optional): Minimum distance between loss and a better loss. Defaults to 0.
        """
        self.patience = patience
        self.delta = delta
        self.wait = 0
        self.mode = "inactive"
        self.counter = 0
    def set_mode(self, mode):
        self.mode = mode
        
    def get_mode(self):
        return self.mode
  
    def track(self, old_loss, new_loss):
        if self.mode == "inactive":
            if new_loss >= old_loss - self.delta:
                if self.counter == 0:
                    self.mode = "active"
                    self.counter = self.patience
                else:
                    self.counter -= 1
            return
    def invest(self, old_loss, new_loss):
        if new_loss < old_loss - self.delta /10:
            self.mode = "inactive"
            return True
        else:
            return False
            
# Copy from stackoverflow
class ProgressBar(object):
    def __init__(self, max_value, disable=True):
        self.max_value = max_value
        self.disable = disable
        self.p = self.pbar()

    def pbar(self):
        return tqdm.tqdm(total=self.max_value,
                         desc='Step: ',
                         disable=self.disable)

    def update(self, update_value):
        self.p.update(update_value)

    def close(self):
        self.p.close()
        
def get_wires_of_gate(gate: typing.Tuple):
    """Get index bit that gate act on

    Args:
        - gate (qiskit.QuantumGate): Quantum gate

    Returns:
        - numpy arrray: list of index bits
    """
    list_wire = []
    for register in gate[1]:
        list_wire.append(register.index)
    return list_wire


def is_gate_in_list_wires(gate: typing.Tuple, wires: typing.List):
    """Check if a gate lies on the next layer or not

    Args:
        - gate (qiskit.QuantumGate): Quantum gate
        - wires (numpy arrray): list of index bits

    Returns:
        - Bool
    """
    list_wire = get_wires_of_gate(gate)
    for wire in list_wire:
        if wire in wires:
            return True
    return False


def split_into_layers(qc: qiskit.QuantumCircuit):
    """Split a quantum circuit into layers

    Args:
        - qc (qiskit.QuantumCircuit): origin circuit

    Returns:
        - list: list of list of quantum gates
    """
    layers = []
    layer = []
    wires = []
    is_param_layer = None
    for gate in qc.data:
        name = gate[0].name
        if name in qtm.constant.ignore_generator:
            continue
        param = gate[0].params
        wire = get_wires_of_gate(gate)
        if is_param_layer is None:
            if len(param) == 0:
                is_param_layer = False
            else:
                is_param_layer = True
        # New layer's condition: depth increase or convert from non-parameterized layer to parameterized layer or vice versa
        if is_gate_in_list_wires(gate, wires) or (is_param_layer == False and len(param) != 0) or (is_param_layer == True and len(param) == 0):
            if is_param_layer == False:
                # First field is 'Is parameterized layer or not?'
                layers.append((False, layer))
            else:
                layers.append((True, layer))
            layer = []
            wires = []
        # Update sub-layer status
        if len(param) == 0 or name == 'state_preparation_dg':
            is_param_layer = False
        else:
            is_param_layer = True
        for w in wire:
            wires.append(w)
        layer.append((name, param, wire))
    # Last sub-layer
    if is_param_layer == False:
        # First field is 'Is parameterized layer or not?'
        layers.append((False, layer))
    else:
        layers.append((True, layer))
    return layers

def create_observers(qc: qiskit.QuantumCircuit, k: int = 0):
    """Return dictionary of observers

    Args:
        - qc (qiskit.QuantumCircuit): Current circuit
        - k (int, optional): Number of observers each layer. Defaults to qc.num_qubits.

    Returns:
        - Dict
    """
    if k == 0:
        k = qc.num_qubits
    observer = []
    for gate in (qc.data)[-k:]:
        gate_name = gate[0].name
        # Non-param gates
        if gate_name in ['barrier', 'swap']:
            continue
        # 2-qubit param gates
        if gate[0].name in ['crx', 'cry', 'crz', 'cx', 'cz']:
            # Take controlled wire as index
            wire = qc.num_qubits - 1 - gate[1][1].index
            # Take control wire as index
            # wire = qc.num_qubits - 1 - gate[1][0].index
        # Single qubit param gates
        else:
            wire = qc.num_qubits - 1 - gate[1][0].index
        observer.append([gate_name, wire])
    return observer

def get_cry_index(qc, thetas):
    """Return a list where i_th = 1 mean thetas[i] is parameter of CRY gate

    Args:
        - func (types.FunctionType): The creating circuit function
        - thetas (np.ndarray): Parameters
    Returns:
        - np.ndarray: The index list has length equal with number of parameters
    """
    layers = split_into_layers(qc)
    index_list = []
    for layer in layers:
        for gate in layer[1]:
            if gate[0] == 'cry':
                index_list.append(1)
            else:
                index_list.append(0)
            if len(index_list) == len(thetas):
                return index_list
    return index_list

def add_layer_into_circuit(qc: qiskit.QuantumCircuit, layer: typing.List):
    """Based on information in layers, adding new gates into current circuit

    Args:
        - qc (qiskit.QuantumCircuit): calculating circuit
        - layer (list): list of gate's informations

    Returns:
        - qiskit.QuantumCircuit: added circuit
    """
    for name, param, wire in layer:
        if name == 'rx':
            qc.rx(param[0], wire[0])
        if name == 'ry':
            qc.ry(param[0], wire[0])
        if name == 'rz':
            qc.rz(param[0], wire[0])
        if name == 'crx':
            qc.crx(param[0], wire[0], wire[1])
        if name == 'cry':
            qc.cry(param[0], wire[0], wire[1])
        if name == 'crz':
            qc.crz(param[0], wire[0], wire[1])
        if name == 'cz':
            qc.cz(wire[0], wire[1])
    return qc

def save_circuit(qc: qiskit.QuantumCircuit, file_name: str) -> None:
    """Save circuit as qpy object

    Args:
        qc (qiskit.QuantumCircuit): Saved circuit
        file_name (str): Path
    """

    with open(f"{file_name}.qpy", "wb") as qpy_file_write:
        qiskit.qpy.dump(qc, qpy_file_write)
    return


def load_circuit(file_name: str) -> qiskit.QuantumCircuit:
    """Load circuit from a specific path

    Args:
        file_name (str): Path

    Returns:
        qiskit.QuantumCircuit
    """
    with open(f"{file_name}.qpy", "rb") as qpy_file_read:
        qc = qiskit.qpy.load(qpy_file_read)[0]
    return qc


def unit_vector(i: int, length: int) -> np.ndarray:
    """Create vector where a[i] = 1 and a[j] = 0 with j <> i

    Args:
        i (int): index
        length (int): dimensional of vector

    Returns:
        np.ndarray
    """
    vector = np.zeros((length))
    vector[i] = 1.0
    return vector






def haar_measure(n):
    """A Random matrix distributed with Haar measure

    Args:
        n (_type_): _description_

    Returns:
        _type_: _description_
    """
    z = (scipy.randn(n, n) + 1j*scipy.randn(n, n))/scipy.sqrt(2.0)
    q, r = scipy.linalg.qr(z)
    d = scipy.diagonal(r)
    ph = d/scipy.absolute(d)
    q = scipy.multiply(q, ph, q)
    return q


def normalize_matrix(matrix):
    """Follow the formula from Bin Ho

    Args:
        matrix (numpy.ndarray): input matrix

    Returns:
        numpy.ndarray: normalized matrix
    """
    return np.conjugate(np.transpose(matrix)) @ matrix / np.trace(np.conjugate(np.transpose(matrix)) @ matrix)

def softmax(xs, scale_max) -> np.ndarray:
    exp_xs = np.exp(xs)
    return 1 + scale_max * exp_xs / np.sum(exp_xs)

    
def is_pos_def(matrix, error=1e-8):
    """_summary_

    Args:
        matrix (_type_): _description_
        error (_type_, optional): _description_. Defaults to 1e-8.

    Returns:
        _type_: _description_
    """
    return np.all(np.linalg.eigvalsh(matrix) > -error)

def is_normalized(matrix):
    """_summary_

    Args:
        matrix (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.isclose(np.trace(matrix), 1)

def truncate_circuit(qc: qiskit.QuantumCircuit, selected_depth: int) -> qiskit.QuantumCircuit:
    """_summary_

    Args:
        qc (qiskit.QuantumCircuit): _description_
        selected_depth (int): _description_

    Returns:
        qiskit.QuantumCircuit: _description_
    """
    if qc.depth() <= selected_depth:
        return qc
    else:
        qc1, qc2 = divide_circuit_by_depth(qc, selected_depth)
        return qc1
    
def calculate_QSP_metric(u: qiskit.QuantumCircuit, vdagger: qiskit.QuantumCircuit, thetas, gibbs=False):
    qc = u.bind_parameters(thetas)
    rho = qiskit.quantum_info.DensityMatrix.from_instruction(qc)
    sigma = qiskit.quantum_info.DensityMatrix.from_instruction(
        vdagger.inverse())
    if gibbs:
        gibbs_rho = qiskit.quantum_info.partial_trace(rho, [0, 1])
        gibbs_sigma = qiskit.quantum_info.partial_trace(sigma, [0, 1])
    else:
        gibbs_rho = None
        gibbs_sigma = None
    trace, fidelity, gibbs_trace, gibbs_trace_fidelity = qtm.utilities.get_metrics(
        rho, sigma, gibbs_rho, gibbs_sigma)
    return trace, np.real(fidelity)


def calculate_QSP_metrics(u: qiskit.QuantumCircuit, vdagger: qiskit.QuantumCircuit, thetass, gibbs=False):
    traces = []
    fidelities = []
    gibbs_traces = []
    gibbs_trace_fidelities = []
    for thetas in thetass:
        # Target state
        # psi = qiskit.quantum_info.Statevector.from_instruction(vdagger).conjugate()
        # rho = qiskit.quantum_info.DensityMatrix(psi)
        # Calculate the metrics
        trace, fidelity, gibbs_trace, gibbs_trace_fidelity = calculate_QSP_metric(
            u, vdagger, thetas, gibbs)
        traces.append(trace)
        fidelities.append(fidelity)
        gibbs_traces.append(gibbs_trace)
        gibbs_trace_fidelities.append(gibbs_trace_fidelity)
    ce = concentratable_entanglement(u.bind_parameters(thetas))
    return traces, fidelities, ce

def divide_circuit(qc: qiskit.QuantumCircuit, percent: float) -> typing.List[qiskit.QuantumCircuit]:
    """Dividing circuit into two sub-circuit

    Args:
        qc (qiskit.QuantumCircuit)
        percent (float): from 0 to 1

    Returns:
        typing.List[qiskit.QuantumCircuit]: two seperated quantum circuits
    """

    qc1 = qiskit.QuantumCircuit(qc.num_qubits)
    qc2 = qc1.copy()
    stop = 0
    for x in qc:
        qc1.append(x[0], x[1])
        stop += 1
        if qc1.depth() / qc.depth() >= percent:
            for x in qc[stop:]:
                qc2.append(x[0], x[1])
            return qc1, qc2
    return qc1, qc2


def divide_circuit_by_depth(qc: qiskit.QuantumCircuit, depth: int) -> typing.List[qiskit.QuantumCircuit]:
    """_Dividing circuit into two sub-circuit

    Args:
        qc (qiskit.QuantumCircuit)
        depth (int): specific depth value

    Returns:
        typing.List[qiskit.QuantumCircuit]: two seperated quantum circuits
    """
    def look_forward(qc, x):
        qc.append(x[0],x[1])
        return qc
    standard_depth = qc.depth()
    qc1 = qiskit.QuantumCircuit(qc.num_qubits)
    qc2 = qc1.copy()
    if depth < 0:
        raise "The depth must be >= 0"
    elif depth == 0:
        qc2 = qc.copy()
    elif depth == standard_depth:
        qc1 = qc.copy()
    else:
        stop = 0
        for i in range(len(qc)):
            qc1.append(qc[i][0], qc[i][1])
            stop += 1
            if qc1.depth() == depth and i + 1 < len(qc) and look_forward(qc1.copy(), qc[i+1]).depth() > depth:
                for x in qc[stop:]:
                    qc2.append(x[0], x[1])
                return qc1, qc2
    return qc1, qc2

def remove_last_gate(qc: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    if qc.data:  
        qc.data.pop()  
    return qc
def compose_circuit(qcs: typing.List[qiskit.QuantumCircuit]) -> qiskit.QuantumCircuit:
    """Combine list of paramerterized quantum circuit into one. It's very helpful!!!

    Args:
        qcs (typing.List[qiskit.QuantumCircuit]): List of quantum circuit

    Returns:
        qiskit.QuantumCircuit: composed quantum circuit
    """
    qc = qiskit.QuantumCircuit(qcs[0].num_qubits)
    i = 0
    num_params = 0
    for sub_qc in qcs:
        num_params += len(sub_qc.parameters)
    thetas = qiskit.circuit.ParameterVector('theta', num_params)
    for sub_qc in qcs:
        for instruction in sub_qc:
            if len(instruction[0].params) == 1:
                instruction[0].params[0] = thetas[i]
                i += 1
            if len(instruction[0].params) == 3:
                instruction[0].params[0] = thetas[i:i+1]
                i += 2
            qc.append(instruction[0], instruction[1])
    qc.draw()
    return qc

# def calculate_QSP_metric(u: qiskit.QuantumCircuit, vdagger: qiskit.QuantumCircuit, thetas, gibbs=False):
#     qc = u.bind_parameters(thetas)
#     rho = qiskit.quantum_info.DensityMatrix.from_instruction(qc)
#     sigma = qiskit.quantum_info.DensityMatrix.from_instruction(
#         vdagger.inverse())
#     if gibbs:
#         gibbs_rho = qiskit.quantum_info.partial_trace(rho, [0, 1])
#         gibbs_sigma = qiskit.quantum_info.partial_trace(sigma, [0, 1])
#     else:
#         gibbs_rho = None
#         gibbs_sigma = None
#     trace, fidelity, gibbs_trace, gibbs_trace_fidelity = qtm.utilities.get_metrics(
#         rho, sigma, gibbs_rho, gibbs_sigma)
#     return trace, np.real(fidelity), gibbs_trace, np.real(gibbs_trace_fidelity)


# def calculate_QST_metric(u: qiskit.QuantumCircuit, vdagger: qiskit.QuantumCircuit, thetas, gibbs=False):
#     rho = qiskit.quantum_info.DensityMatrix.from_instruction(u)
#     qc = vdagger.bind_parameters(thetas).inverse()
#     sigma = qiskit.quantum_info.DensityMatrix.from_instruction(qc)
#     if gibbs:
#         gibbs_rho = qiskit.quantum_info.partial_trace(rho, [0, 1])
#         gibbs_sigma = qiskit.quantum_info.partial_trace(sigma, [0, 1])
#     else:
#         gibbs_rho = None
#         gibbs_sigma = None
#     trace, fidelity, gibbs_trace, gibbs_trace_fidelity = qtm.utilities.get_metrics(
#         rho, sigma, gibbs_rho, gibbs_sigma)
#     return trace, np.real(fidelity), gibbs_trace, np.real(gibbs_trace_fidelity)


# def calculate_QSP_metrics(u: qiskit.QuantumCircuit, vdagger: qiskit.QuantumCircuit, thetass, gibbs=False):
#     traces = []
#     fidelities = []
#     gibbs_traces = []
#     gibbs_trace_fidelities = []
#     for thetas in thetass:
#         # Target state
#         # psi = qiskit.quantum_info.Statevector.from_instruction(vdagger).conjugate()
#         # rho = qiskit.quantum_info.DensityMatrix(psi)
#         # Calculate the metrics
#         trace, fidelity, gibbs_trace, gibbs_trace_fidelity = calculate_QSP_metric(
#             u, vdagger, thetas, gibbs)
#         traces.append(trace)
#         fidelities.append(fidelity)
#         gibbs_traces.append(gibbs_trace)
#         gibbs_trace_fidelities.append(gibbs_trace_fidelity)
#     ce = concentratable_entanglement(u.bind_parameters(thetas))
#     return traces, fidelities, gibbs_traces, gibbs_trace_fidelities, ce


# def calculate_QST_metrics(u: qiskit.QuantumCircuit, vdagger: qiskit.QuantumCircuit, thetass, gibbs=False):
#     traces = []
#     fidelities = []
#     gibbs_traces = []
#     gibbs_trace_fidelities = []
#     for thetas in thetass:
#         trace, fidelity, gibbs_trace, gibbs_trace_fidelity = calculate_QST_metric(
#             u, vdagger, thetas, gibbs)
#         traces.append(trace)
#         fidelities.append(fidelity)
#         gibbs_traces.append(gibbs_trace)
#         gibbs_trace_fidelities.append(gibbs_trace_fidelity)

#     ce = concentratable_entanglement(vdagger.bind_parameters(thetas))
#     return traces, fidelities, gibbs_traces, gibbs_trace_fidelities, ce

