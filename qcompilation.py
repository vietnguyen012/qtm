import qtm.measure
import qtm.gradient
import qtm.optimizer
import qtm.loss
import qtm.utilities
import qtm.metric
import qtm.qst
import qtm.qsp
import numpy as np
import typing
import types
import qiskit
import matplotlib.pyplot as plt


class QuantumCompilation():
    def __init__(self) -> None:
        self.u = None
        self.vdagger = None
        self.is_trained = False
        self.optimizer = None
        self.loss_func = None
        self.thetas = None
        self.thetass = []
        self.loss_values = []
        self.compilation_fidelities = []
        self.compilation_traces = []
        self.gibbs_fidelities = []
        self.gibbs_traces = []
        self.ces = None
        self.kwargs = None
        self.is_evolutional = False
        self.num_steps = 0
        self.gibbs = False
        return

    def __init__(self, u: qiskit.QuantumCircuit, vdagger: qiskit.QuantumCircuit, optimizer: typing.Union[types.FunctionType, str], loss_func: typing.Union[types.FunctionType, str], thetas: np.ndarray = np.array([]), **kwargs):
        """_summary_

        Args:
            - u (qiskit.QuantumCircuit]): In quantum state preparation problem, this is the ansatz. In tomography, this is the circuit that generate random Haar state.
            - vdagger (qiskit.QuantumCircuit]): In quantum tomography problem, this is the ansatz. In state preparation, this is the circuit that generate random Haar state.
            - optimizer (typing.Union[types.FunctionType, str]): You can put either string or function here. If type string, qcompilation produces some famous optimizers such as: 'sgd', 'adam', 'qng-fubini-study', 'qng-qfim', 'qng-adam'.
            - loss_func (typing.Union[types.FunctionType, str]): You can put either string or function here. If type string, qcompilation produces some famous optimizers such as: 'loss_basic'  (1 - p0) and 'loss_fubini_study' (\sqrt{(1 - p0)}).
            - thetas (np.ndarray, optional): initial parameters. Note that it must fit with your ansatz. Defaults to np.array([]).
        """
        self.set_u(u)
        self.set_vdagger(vdagger)
        self.set_optimizer(optimizer)
        self.set_loss_func(loss_func)
        self.set_kwargs(**kwargs)
        self.set_thetas(thetas)
        self.thetass = []
        self.loss_values = []
        self.compilation_fidelities = []
        self.compilation_traces = []
        self.gibbs_fidelities = []
        self.gibbs_traces = []
        self.ces = None
        self.kwargs = None
        self.is_evolutional = False
        self.num_steps = 0
        self.gibbs = False
        return

    def set_u(self, _u: qiskit.QuantumCircuit):
        """In quantum state preparation problem, this is the ansatz. In tomography, this is the circuit that generate random Haar state.

        Args:
            - _u (typing.Union[types.FunctionType, qiskit.QuantumCircuit]): init circuit
        """
        if isinstance(_u, qiskit.QuantumCircuit):
            self.u = _u
        else:
            raise ValueError('The U part must be a determined quantum circuit')
        return

    def set_vdagger(self, _vdagger):
        """In quantum state tomography problem, this is the ansatz. In state preparation, this is the circuit that generate random Haar state.

        Args:
            - _vdagger (qiskit.QuantumCircuit): init circuit
        """
        if isinstance(_vdagger, qiskit.QuantumCircuit):
            self.vdagger = _vdagger
        else:
            raise ValueError(
                'The V dagger part must be a determined quantum circuit')
        return

    def set_loss_func(self, _loss_func: typing.Union[types.FunctionType, str]):
        """Set the loss function for compiler

        Args:
            - _loss_func (typing.Union[types.FunctionType, str])

        Raises:
            ValueError: when you pass wrong type
        """
        if callable(_loss_func):
            self.loss_func = _loss_func
        elif isinstance(_loss_func, str):
            if _loss_func == 'loss_basic':
                self.loss_func = qtm.loss.loss_basis
            elif _loss_func == 'loss_fubini_study':
                self.loss_func = qtm.loss.loss_fubini_study
        else:
            raise ValueError(
                'The loss function must be a function f: measurement value -> loss value or string in ["loss_basic", "loss_fubini_study"]')
        return

    def set_optimizer(self, _optimizer: typing.Union[types.FunctionType, str]):
        """Change the optimizer of the compiler

        Args:
            - _optimizer (typing.Union[types.FunctionType, str])

        Raises:
            ValueError: when you pass wrong type
        """
        if callable(_optimizer):
            self.optimizer = _optimizer
        elif isinstance(_optimizer, str):
            if _optimizer == 'sgd':
                self.optimizer = qtm.optimizer.sgd
            elif _optimizer == 'adam':
                self.optimizer = qtm.optimizer.adam
            elif _optimizer == 'qng_fubini_study':
                self.optimizer = qtm.optimizer.qng_fubini_study
            elif _optimizer == 'qng_fubini_study_hessian':
                self.optimizer = qtm.optimizer.qng_fubini_study_hessian
            elif _optimizer == 'qng_fubini_study_scheduler':
                self.optimizer = qtm.optimizer.qng_fubini_study_scheduler
            elif _optimizer == 'qng_qfim':
                self.optimizer = qtm.optimizer.qng_qfim
            elif _optimizer == 'qng_adam':
                self.optimizer = qtm.optimizer.qng_adam
        else:
            raise ValueError(
                'The optimizer must be a function f: thetas -> thetas or string in ["sgd", "adam", "qng_qfim", "qng_fubini_study", "qng_adam"]')
        return

    def set_num_steps(self, _num_steps: int):
        """Set the number of iteration for compiler

        Args:
            - _num_steps (int): number of iterations

        Raises:
            ValueError: when you pass a nasty value
        """
        if _num_steps > 0 and isinstance(_num_steps, int):
            self.num_steps = _num_steps
        else:
            raise ValueError(
                'Number of iterations must be an integer, take example: 10 or 100.')
        return

    def set_thetas(self, _thetas: np.ndarray):
        """Set parameter, it will be updated at each iteration

        Args:
            _thetas (np.ndarray): parameter for u or vdagger
        """
        if isinstance(_thetas, np.ndarray):
            self.thetas = _thetas
        else:
            raise ValueError('The parameter must be numpy array')
        return

    def set_kwargs(self, **kwargs):
        """Arguments supported for u or vdagger only. Ex: number of layer
        """
        self.__dict__.update(**kwargs)
        self.kwargs = kwargs
        return

    def fit(self, num_steps: int = 100, metrics: typing.List[str] = 'compilation', verbose: int = 0):
        """Optimize the thetas parameters

        Args:
            - num_steps: number of iterations
            - verbose (int, optional): 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per 10 steps. Verbose 1 is good for timing training time, verbose 2 if you want to log loss values to a file. Please install package tdqm if you want to use verbose 1. 
            - metrics (List[str]): list of metric name that you want, take example, ['compilation', 'gibbs']
        """
        self.num_steps = num_steps
        if len(self.thetas) == 0:
            if (len(self.u.parameters)) > 0:
                self.thetas = np.ones(len(self.u.parameters))
            else:
                self.thetas = np.ones(len(self.vdagger.parameters))
        self.is_trained = True
        
        """Return the new thetas that fit with the circuit from create_u_func function

        Args:
            u (qiskit.QuantumCircuit): _description_
            vdagger (qiskit.QuantumCircuit): _description_
            thetas (np.ndarray): _description_
            num_steps (int): _description_
            loss_func (types.FunctionType): _description_
            optimizer (types.FunctionType): _description_
            verbose (int, optional): _description_. Defaults to 0.
            is_return_all_thetas (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if verbose == 1:
            bar = qtm.utilities.ProgressBar(max_value=num_steps, disable=False)
        uvaddager = self.u.compose(self.vdagger)
        for i in range(0, num_steps):
            grad_loss = qtm.gradient.grad_loss(uvaddager, self.thetas)
            optimizer_name = self.optimizer.__name__

            if optimizer_name == 'sgd':
                self.thetas = qtm.optimizer.sgd(self.thetas, grad_loss)

            elif optimizer_name == 'adam':
                if i == 0:
                    m, v1 = list(np.zeros(self.thetas.shape[0])), list(
                        np.zeros(self.thetas.shape[0]))
                self.thetas = qtm.optimizer.adam(self.thetas, m, v1, i, grad_loss)

            elif 'qng' in optimizer_name:
                grad_psi1 = qtm.measure.grad_psi(uvaddager, self.thetas,
                                            r=1 / 2,
                                            s=np.pi)
                qc_binded = uvaddager.bind_parameters(self.thetas)
                psi = qiskit.quantum_info.Statevector.from_instruction(qc_binded).data
                psi = np.expand_dims(psi, 1)
                if optimizer_name == 'qng_fubini_study':
                    G = qtm.gradient.qng(uvaddager)
                    self.thetas = qtm.optimizer.qng_fubini_study(thetas, G, grad_loss)
                if optimizer_name == 'qng_fubini_hessian':
                    G = qtm.gradient.qng_hessian(uvaddager)
                    self.thetas = qtm.optimizer.qng_fubini_study(self.thetas, G, grad_loss)
                if optimizer_name == 'qng_fubini_study_scheduler':
                    G = qtm.gradient.qng(uvaddager)
                    self.thetas = qtm.optimizer.qng_fubini_study_scheduler(
                        self.thetas, G, grad_loss, i)
                if optimizer_name == 'qng_qfim':

                    self.thetas = qtm.optimizer.qng_qfim(
                        self.thetas, psi, grad_psi1, grad_loss)

                if optimizer_name == 'qng_adam':
                    if i == 0:
                        m, v1 = list(np.zeros(thetas.shape[0])), list(
                            np.zeros(thetas.shape[0]))
                    self.thetas = qtm.optimizer.qng_adam(
                        self.thetas, m, v1, i, psi, grad_psi1, grad_loss)
            else:
                thetas = self.optimizer(self.thetas, grad_loss)
            
            qc_binded = uvaddager.bind_parameters(self.thetas)
            loss = self.loss_func(
                qtm.measure.measure(qc_binded, list(range(self.u.num_qubits))))
            self.loss_values.append(loss)
            self.thetass.append(self.thetas.copy())
            if verbose == 1:
                bar.update(1)
            if verbose == 2 and i % 10 == 0:
                print("Step " + str(i) + ": " + str(loss))

        if verbose == 1:
            bar.close()

        metric_params = [self.u, self.vdagger, self.thetass]
        if 'compilation' in metrics:
            self.compilation_traces, self.compilation_fidelities = qtm.metric.calculate_compilation_metrics(*metric_params)
        if 'gibbs' in metrics:
            self.gibbs_traces, self.gibbs_fidelities = qtm.metric.calculate_gibbs_metrics(*metric_params)
        if 'ce' in metrics:
            self.ces = qtm.metric.calculate_ce_metrics(*metric_params)
        return

    def plot(self):
        plt.plot(self.loss_values)
        plt.ylabel("Loss values")
        plt.xlabel('Num. iteration')
        return
    
    def plot_animation(self, interval: int = 100, file_name: str = 'test.gif'):
        import matplotlib.animation as animation
        x = np.linspace(0, int(self.num_steps), int(self.num_steps))
        y1 = self.loss_values
        y2 = self.compilation_fidelities
        y3 = self.compilation_traces
        fig, ax = plt.subplots()
        ax.set_xlim(int(-self.num_steps*0.05), int(self.num_steps*1.05))
        ax.set_ylim(-0.05, 1.05)
        loss_text = ax.text(0, 0, "", fontsize=12)
        fid_text = ax.text(0, 0, "", fontsize=12)
        trace_text = ax.text(0, 0, "", fontsize=12)
        plt.ylabel("Loss values")
        plt.xlabel('Num. iteration')
        xs = []
        ys1, ys2, ys3 = [], [], []
        def update(i):
            xs.append(x[i])
            ys1.append(y1[i])
            ys2.append(y2[i])
            ys3.append(y3[i])
            ax.plot(xs, ys1, color='blue', label = 'Loss value')
            ax.plot(xs, ys2, color='red', label = 'Fidelity')
            ax.plot(xs, ys3, color='green', label = 'Trace')
            loss_text.set_position([xs[i], ys1[i]])
            loss_text.set_text('Loss: ' + str(np.round(ys1[i], 2)))
            fid_text.set_position([xs[i], ys2[i]])
            fid_text.set_text('Fidelity: ' + str(np.round(ys2[i], 2)))
            trace_text.set_position([xs[i], ys3[i]])
            trace_text.set_text('Trace: ' + str(np.round(ys3[i], 2)))
        animator = animation.FuncAnimation(fig, update,
                                    interval=interval, repeat=False)
        animator.save(file_name)
        
    def save(self, ansatz, state, file_name):
        if (len(self.u.parameters)) > 0:
            qspobj = qtm.qsp.QuantumStatePreparation(
                self.u,
                self.vdagger,
                self.thetas,
                ansatz)
            qspobj.save(state, file_name)
        else:
            qstobj = qtm.qst.QuantumStateTomography(
                self.u,
                self.vdagger,
                self.thetas,
                ansatz)
            qstobj.save(state, file_name)
        return

    def reset(self):
        """Delete all current property of compiler
        """
        self.u = None
        self.vdagger = None
        self.is_trained = False
        self.optimizer = None
        self.loss_func = None
        self.num_steps = 0
        self.thetas = None
        self.thetass = []
        self.loss_values = []
        return
