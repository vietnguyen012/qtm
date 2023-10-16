
import qtm.ansatz
import qtm.constant
import typing
import numpy as np
import qiskit
import scipy

def calculate_ce_metric(u: qiskit.QuantumCircuit, exact=False):
    """calculate_concentratable_entanglement

    Args:
        u (qiskit.QuantumCircuit): _description_
        exact (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    qubit = list(range(u.num_qubits))
    n = len(qubit)
    cbits = qubit.copy()
    swap_test_circuit = qtm.ansatz.parallized_swap_test(u)

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


def compilation_trace_distance(rho, sigma):
    """Since density matrices are Hermitian, so trace distance is 1/2 (Sigma(|lambdas|)) with lambdas are the eigenvalues of (rho_psi - rho_psi_hat) matrix

    Args:
        - rho (DensityMatrix): first density matrix
        - sigma (DensityMatrix): second density matrix

    Returns:
        - float: trace metric has value from 0 to 1
    """
    w, _ = np.linalg.eig((rho - sigma).data)
    return 1 / 2 * sum(abs(w))


def compilation_trace_fidelity(rho, sigma):
    """Calculating the fidelity metric

    Args:
        - rho (DensityMatrix): first density matrix
        - sigma (DensityMatrix): second density matrix

    Returns:
        - float: trace metric has value from 0 to 1
    """
    rho = rho.data
    sigma = sigma.data
    return np.real(np.trace(
        scipy.linalg.sqrtm(
            (scipy.linalg.sqrtm(rho)) @ (rho)) @ (scipy.linalg.sqrtm(sigma))))


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


def gibbs_trace_distance(rho):
    """_summary_

    Args:
        rho (_type_): _description_

    Returns:
        _type_: _description_
    """
    if rho is None:
        return None
    return np.trace(np.linalg.matrix_power(rho, 2))


def calculate_premetric(u: qiskit.QuantumCircuit, vdagger: qiskit.QuantumCircuit, thetas):
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
        sigma = qiskit.quantum_info.DensityMatrix.from_instruction(vdagger.inverse())
        
    else:
        qc = vdagger.bind_parameters(thetas).inverse()
        rho = qiskit.quantum_info.DensityMatrix.from_instruction(u)
        sigma = qiskit.quantum_info.DensityMatrix.from_instruction(qc)
    return qc, rho, sigma


def calculate_gibbs_metrics(u: qiskit.QuantumCircuit, vdagger: qiskit.QuantumCircuit, thetass):
    """_summary_

    Args:
        u (qiskit.QuantumCircuit): _description_
        vdagger (qiskit.QuantumCircuit): _description_
        thetass (_type_): _description_
        gibbs (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    gibbs_traces = []
    gibbs_fidelities = []
    for thetas in thetass:
        _, rho, sigma = calculate_premetric(u, vdagger, thetas)
        gibbs_rho = qiskit.quantum_info.partial_trace(rho, [0, 1])
        gibbs_sigma = qiskit.quantum_info.partial_trace(sigma, [0, 1])
        gibbs_trace = gibbs_trace_distance(gibbs_rho)
        gibbs_fidelity = gibbs_trace_fidelity(gibbs_rho, gibbs_sigma)
        gibbs_traces.append(gibbs_trace)
        gibbs_fidelities.append(gibbs_fidelity)
    return gibbs_traces, gibbs_fidelities

def calculate_ce_metrics(u: qiskit.QuantumCircuit, vdagger: qiskit.QuantumCircuit, thetass):
    ces = []
    for thetas in thetass:
        qc, _, _ = calculate_premetric(u, vdagger, thetas)
        ces.append(calculate_ce_metric(qc))
    return ces
    
def calculate_compilation_metrics(u: qiskit.QuantumCircuit, vdagger: qiskit.QuantumCircuit, thetass):
    """_summary_

    Args:
        u (qiskit.QuantumCircuit): _description_
        vdagger (qiskit.QuantumCircuit): _description_
        thetass (_type_): _description_
        gibbs (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    compilation_traces = []
    compilation_fidelities = []
    for thetas in thetass:
        _, rho, sigma = calculate_premetric(u, vdagger, thetas)
        compilation_trace = compilation_trace_distance(rho, sigma)
        compilation_fidelity = compilation_trace_fidelity(rho, sigma)
        compilation_traces.append(compilation_trace)
        compilation_fidelities.append(compilation_fidelity)
    return compilation_traces, compilation_fidelities