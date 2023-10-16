
import qiskit
import numpy as np
import qtm.constant
import qtm.utilities
import qtm.measure
import typing
import scipy


def single_2term_psr(qc: qiskit.QuantumCircuit, thetas: np.ndarray, i, **kwargs):
    thetas1, thetas2 = thetas.copy(), thetas.copy()
    thetas1[i] += qtm.constant.two_term_psr['s']
    thetas2[i] -= qtm.constant.two_term_psr['s']

    qc1 = qc.bind_parameters(thetas1)
    qc2 = qc.bind_parameters(thetas2)
    return -qtm.constant.two_term_psr['r'] * (
        qtm.measure.measure(qc1, list(range(qc1.num_qubits))) -
        qtm.measure.measure(qc2, list(range(qc2.num_qubits))))


def single_4term_psr(qc: qiskit.QuantumCircuit, thetas: np.ndarray, i, **kwargs):
    thetas1, thetas2 = thetas.copy(), thetas.copy()
    thetas3, thetas4 = thetas.copy(), thetas.copy()
    thetas1[i] += qtm.constant.four_term_psr['alpha']
    thetas2[i] -= qtm.constant.four_term_psr['alpha']
    thetas3[i] += qtm.constant.four_term_psr['beta']
    thetas4[i] -= qtm.constant.four_term_psr['beta']
    qc1 = qc.bind_parameters(thetas1)
    qc2 = qc.bind_parameters(thetas2)
    qc3 = qc.bind_parameters(thetas3)
    qc4 = qc.bind_parameters(thetas4)
    return - (qtm.constant.four_term_psr['d_plus'] * (
        qtm.measure.measure(qc1, list(range(qc1.num_qubits))) -
        qtm.measure.measure(qc2, list(range(qc2.num_qubits)))) - qtm.constant.four_term_psr['d_minus'] * (
        qtm.measure.measure(qc3, list(range(qc3.num_qubits))) -
        qtm.measure.measure(qc4, list(range(qc4.num_qubits)))))


def grad_loss(qc: qiskit.QuantumCircuit, thetas: np.ndarray):
    """Return the gradient of the loss function

    L = 1 - |<psi~|psi>|^2 = 1 - P_0

    => nabla_L = - nabla_P_0 = - r (P_0(+s) - P_0(-s))

    Args:
        - qc (QuantumCircuit): Parameterized quantum circuit 
        - thetas (np.ndarray): Parameters

    Returns:
        - np.ndarray: the gradient vector
    """
    index_list = qtm.utilities.get_cry_index(qc, thetas)
    grad_loss = np.zeros(len(thetas))

    for i in range(0, len(thetas)):
        if index_list[i] == 0:
            # In equation (13)
            grad_loss[i] = single_2term_psr(qc, thetas, i)
        if index_list[i] == 1:
            # In equation (14)
            grad_loss[i] = single_4term_psr(qc, thetas, i)
    return grad_loss


def grad_psi(qc: qiskit.QuantumCircuit, thetas: np.ndarray, r: float, s: float):
    """Return the derivatite of the psi base on parameter shift rule

    Args:
        - qc (qiskit.QuantumCircuit): Parameterized quantum circuit 
        - thetas (np.ndarray): parameters
        - r (float): in psr formula
        - s (float): in psr formula

    Returns:
        - np.ndarray: N x N matrix
    """
    gradient_psi = []
    for i in range(0, len(thetas)):
        thetas_copy = thetas.copy()
        thetas_copy[i] += s
        qc_copy = qc.bind_parameters(thetas_copy)
        psi_qc = qiskit.quantum_info.Statevector.from_instruction(qc_copy).data
        psi_qc = np.expand_dims(psi_qc, 1)
        gradient_psi.append(r * psi_qc)
    gradient_psi = np.array(gradient_psi)
    return gradient_psi


def qfim(psi: np.ndarray, grad_psi: np.ndarray):
    """Create Quantum Fisher Information matrix base on 
    \n https://quantum-journal.org/views/qv-2021-10-06-61/

    Args:
        - psi (np.ndarray): current state vector, is a N x 1 matrix
        - grad_psi (np.ndarray): all partial derivatives of $\psi$, is a N x N matrix

    Returns:
        np.ndarray: N x N matrix
    """
    num_params = grad_psi.shape[0]
    # Calculate elements \bra\psi|\partial_k \psi\ket
    F_elements = np.zeros(num_params, dtype=np.complex128)
    for i in range(num_params):
        F_elements[i] = np.transpose(np.conjugate(psi)) @ (grad_psi[i])
    # Calculate F[i, j] = 4*Re*[\bra\partial_i \psi | \partial_j \psi \ket -
    # \bra\partial_i\psi | \psi\ket * \bra\psi|\partial_j \psi\ket]
    F = np.zeros([num_params, num_params])
    for i in range(0, num_params):
        for j in range(0, num_params):
            F[i, j] = 4 * np.real(
                np.transpose(np.conjugate(grad_psi[i])) @ (grad_psi[j]) -
                np.transpose(np.conjugate(F_elements[i])) * (F_elements[j]))
            if F[i, j] < 10**(-15):
                F[i, j] = 0
    return F




def calculate_g(qc: qiskit.QuantumCircuit, observers: typing.Dict[str, int]):
    """Fubini-Study tensor. Detail informations: 
    \n https://pennylane.ai/qml/demos/tutorial_quantum_natural_gradient.html

    Args:
        - qc (qiskit.QuantumCircuit): Current quantum circuit
        - observers (Dict[str, int]): List of observer type and its acting wire

    Returns:
        - numpy array: block-diagonal submatrix g
    """
    # Get |psi>
    psi = qiskit.quantum_info.Statevector.from_instruction(qc).data
    psi = np.expand_dims(psi, 1)
    # Get <psi|
    psi_hat = np.transpose(np.conjugate(psi))
    num_observers = len(observers)
    num_qubits = qc.num_qubits
    g = np.zeros([num_observers, num_observers], dtype=np.complex128)
    # Each K[j] must have 2^n x 2^n dimensional with n is the number of qubits
    Ks = []
    # Observer shorts from high to low
    for observer_name, observer_wire in observers:
        observer = qtm.constant.generator[observer_name]
        if observer_wire == 0:
            K = observer
        else:
            if observer_name in ['crx', 'cry', 'crz', 'cz']:
                K = qtm.constant.generator['11']
            else:
                K = qtm.constant.generator['i']
        for i in range(1, num_qubits):
            if i == observer_wire:
                K = np.kron(K, observer)
            else:
                if observer_name in ['crx', 'cry', 'crz', 'cz']:
                    K = np.kron(K, qtm.constant.generator['11'])
                else:
                    K = np.kron(K, qtm.constant.generator['i'])
        Ks.append(K)
    for i in range(0, num_observers):
        for j in range(0, num_observers):
            g[i, j] = psi_hat @ (Ks[i] @ Ks[j]) @ psi - (
                psi_hat @ Ks[i] @ psi) * (psi_hat @ Ks[j] @ psi)
            if g[i, j] < 10**(-10):
                g[i, j] = 0
    return g


######################################
## General quantum natural gradient ##
######################################


def qng_hessian(uvdagger: qiskit.QuantumCircuit, thetas: np.ndarray):
    """_summary_

    Args:
        uvdagger (qiskit.QuantumCircuit): _description_
        thetas (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    alpha = 0.01
    length = thetas.shape[0]
    thetas_origin = thetas

    def f(thetas):
        qc = uvdagger.bind_parameters(thetas)
        qc_reverse = uvdagger.bind_parameters(thetas_origin).inverse()
        qc = qc.compose(qc_reverse)
        return qtm.measure.measure(qc, list(range(qc.num_qubits)))
    G = [[0 for _ in range(length)] for _ in range(length)]
    for i in range(0, length):
        for j in range(0, length):
            k1 = f(thetas + alpha*(qtm.utilities.unit_vector(i,
                   length) + qtm.utilities.unit_vector(j, length)))
            k2 = -f(thetas + alpha * (qtm.utilities.unit_vector(i,
                    length) - qtm.utilities.unit_vector(j, length)))
            k3 = -f(thetas - alpha * (qtm.utilities.unit_vector(i,
                    length) - qtm.utilities.unit_vector(j, length)))
            k4 = f(thetas - alpha*(qtm.utilities.unit_vector(i,
                   length) + qtm.utilities.unit_vector(j, length)))
            G[i][j] = (1/(4*(np.sin(alpha))**2))*(k1 + k2 + k3 + k4)
    return -1/2*np.asarray(G)


def qng(uvaddger: qiskit.QuantumCircuit):
    """Calculate G matrix in qng

    Args:
        - qc (qiskit.QuantumCircuit)
        - thetas (np.ndarray): parameters
        - create_circuit_func (FunctionType)
        - num_layers (int): number of layer of ansatz

    Returns:
        - np.ndarray: G matrix
    """
    n = uvaddger.num_qubits
    # List of g matrices
    gs = []
    # Splitting circuit into list of V and W sub-layer (non-parameter and parameter)
    layers = qtm.utilities.split_into_layers(uvaddger)
    qc = qiskit.QuantumCircuit(n, n)
    for is_param_layer, layer in layers:
        if is_param_layer:
            observers = qtm.utilities.create_observers(
                qtm.utilities.add_layer_into_circuit(qc.copy(), layer), len(layer))
            gs.append(calculate_g(qc, observers))
        # Add next sub-layer into the current circuit
        qc = qtm.utilities.add_layer_into_circuit(qc, layer)

    G = gs[0]
    for i in range(1, len(gs)):
        G = scipy.linalg.block_diag(G, gs[i])
    return G
