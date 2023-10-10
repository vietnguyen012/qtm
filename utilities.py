import typing
import numpy as np
import qiskit
import scipy
import qtm.constant



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


def parallized_swap_test(u: qiskit.QuantumCircuit):
    """_summary_

    Args:
        u (qiskit.QuantumCircuit): _description_

    Returns:
        _type_: _description_
    """
    # circuit = qtm.state.create_w_state(5)
    n_qubit = u.num_qubits
    qubits_list_first = list(range(n_qubit, 2*n_qubit))
    qubits_list_second = list(range(2*n_qubit, 3*n_qubit))

    # Create swap test circuit
    swap_test_circuit = qiskit.QuantumCircuit(3*n_qubit, n_qubit)

    # Add initial circuit the first time

    swap_test_circuit = swap_test_circuit.compose(u, qubits=qubits_list_first)
    # Add initial circuit the second time
    swap_test_circuit = swap_test_circuit.compose(u, qubits=qubits_list_second)
    swap_test_circuit.barrier()

    # Add hadamard gate
    swap_test_circuit.h(list(range(0, n_qubit)))
    swap_test_circuit.barrier()

    for i in range(n_qubit):
        # Add control-swap gate
        swap_test_circuit.cswap(i, i+n_qubit, i+2*n_qubit)
    swap_test_circuit.barrier()

    # Add hadamard gate
    swap_test_circuit.h(list(range(0, n_qubit)))
    swap_test_circuit.barrier()
    return swap_test_circuit


def concentratable_entanglement(u: qiskit.QuantumCircuit, exact=False):
    """_summary_

    Args:
        u (qiskit.QuantumCircuit): _description_
        exact (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    qubit = list(range(u.num_qubits))
    n = len(qubit)
    cbits = qubit.copy()
    swap_test_circuit = parallized_swap_test(u)

    if exact:
        statevec = qiskit.quantum_info.Statevector(swap_test_circuit)
        statevec.seed(value=42)
        probs = statevec.evolve(
            swap_test_circuit).probabilities_dict(qargs=qubit)
        return 1 - probs["0"*len(qubit)]
    else:
        for i in range(0, n):
            swap_test_circuit.measure(qubit[i], cbits[i])

        counts = qiskit.execute(
            swap_test_circuit, backend=qtm.constant.backend, shots=qtm.constant.NUM_SHOTS
        ).result().get_counts()

        return 1-counts.get("0"*len(qubit), 0)/qtm.constant.NUM_SHOTS


def extract_state(qc: qiskit.QuantumCircuit):
    """Get infomation about quantum circuit

    Args:
        - qc (qiskit.QuantumCircuit): Extracted circuit

    Returns:
       - tuple: state vector and density matrix
    """
    psi = qiskit.quantum_info.Statevector.from_instruction(qc)
    rho_psi = qiskit.quantum_info.DensityMatrix(psi)
    return psi, rho_psi


def trace_distance(rho, sigma):
    """Since density matrices are Hermitian, so trace distance is 1/2 (Sigma(|lambdas|)) with lambdas are the eigenvalues of (rho_psi - rho_psi_hat) matrix

    Args:
        - rho (DensityMatrix): first density matrix
        - sigma (DensityMatrix): second density matrix

    Returns:
        - float: trace metric has value from 0 to 1
    """
    w, _ = np.linalg.eig((rho - sigma).data)
    return 1 / 2 * sum(abs(w))


def trace_fidelity(rho, sigma):
    """Calculating the fidelity metric

    Args:
        - rho (DensityMatrix): first density matrix
        - sigma (DensityMatrix): second density matrix

    Returns:
        - float: trace metric has value from 0 to 1
    """
    rho = rho.data
    sigma = sigma.data
    return np.trace(
        scipy.linalg.sqrtm(
            (scipy.linalg.sqrtm(rho)) @ (rho)) @ (scipy.linalg.sqrtm(sigma)))


def gibbs_trace_fidelity(rho, sigma):
    """_summary_

    Args:
        rho (_type_): _description_
        sigma (_type_): _description_

    Returns:
        _type_: _description_
    """
    if rho is None:
        return None
    half_power_sigma = scipy.linalg.fractional_matrix_power(sigma, 1/2)
    return np.trace(scipy.linalg.sqrtm(
        half_power_sigma @ rho.data @ half_power_sigma))


def gibbs_trace(rho):
    """_summary_

    Args:
        rho (_type_): _description_

    Returns:
        _type_: _description_
    """
    if rho is None:
        return None
    return np.trace(np.linalg.matrix_power(rho, 2))


def get_metrics(rho, sigma, gibbs_rho, gibbs_sigma):
    """Get different metrics between the origin state and the reconstructed state

    Args:
        - psi (Statevector): first state vector
        - sigma (Statevector): second state vector

    Returns:
        - Tuple: trace and fidelity between two input vectors
    """
    return qtm.utilities.trace_distance(rho, sigma), qtm.utilities.trace_fidelity(rho, sigma), qtm.utilities.gibbs_trace(
                                            gibbs_rho), qtm.utilities.gibbs_trace_fidelity(gibbs_rho, gibbs_sigma)


def calculate_metric(u: qiskit.QuantumCircuit, vdagger: qiskit.QuantumCircuit, thetas, gibbs=False):
    """_summary_

    Args:
        u (qiskit.QuantumCircuit): _description_
        vdagger (qiskit.QuantumCircuit): _description_
        thetas (_type_): _description_
        gibbs (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if (len(u.parameters)) > 0:
        qc = u.bind_parameters(thetas)
        rho = qiskit.quantum_info.DensityMatrix.from_instruction(qc)
        sigma = qiskit.quantum_info.DensityMatrix.from_instruction(
            vdagger.inverse())
        ce = concentratable_entanglement(u.bind_parameters(thetas))
    else:
        rho = qiskit.quantum_info.DensityMatrix.from_instruction(u)
        qc = vdagger.bind_parameters(thetas).inverse()
        sigma = qiskit.quantum_info.DensityMatrix.from_instruction(qc)
        ce = concentratable_entanglement(vdagger.bind_parameters(thetas))
    if gibbs:
        gibbs_rho = qiskit.quantum_info.partial_trace(rho, [0, 1])
        gibbs_sigma = qiskit.quantum_info.partial_trace(sigma, [0, 1])
    else:
        gibbs_rho = None
        gibbs_sigma = None
    trace, fidelity, gibbs_trace, gibbs_trace_fidelity = get_metrics(
        rho, sigma, gibbs_rho, gibbs_sigma)
    return trace, np.real(fidelity), gibbs_trace, np.real(gibbs_trace_fidelity), ce 

def calculate_metrics(u: qiskit.QuantumCircuit, vdagger: qiskit.QuantumCircuit, thetass, gibbs=False):
    """_summary_

    Args:
        u (qiskit.QuantumCircuit): _description_
        vdagger (qiskit.QuantumCircuit): _description_
        thetass (_type_): _description_
        gibbs (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    traces = []
    fidelities = []
    gibbs_traces = []
    gibbs_trace_fidelities = []
    ces = []
    for thetas in thetass:
        trace, fidelity, gibbs_trace, gibbs_trace_fidelity, ce = calculate_metric(
            u, vdagger, thetas, gibbs)
        traces.append(trace)
        fidelities.append(fidelity)
        gibbs_traces.append(gibbs_trace)
        gibbs_trace_fidelities.append(gibbs_trace_fidelity)
        ces.append(ce)
    return traces, fidelities, gibbs_traces, gibbs_trace_fidelities, ces

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

