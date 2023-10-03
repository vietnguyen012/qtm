import qtm.base
import qtm.optimizer
import qtm.loss
import qtm.utilities
import qtm.qst
import qtm.qsp
import numpy as np
import typing
import types
import qiskit
import matplotlib.pyplot as plt
import pickle


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
        self.fidelities = []
        self.traces = []
        self.ce = None
        self.kwargs = None
        self.is_evolutional = False
        self.num_steps = 0
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
                'Number of iterations must be a integer, such that 10 or 100.')
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

    def fit(self, num_steps: int = 100, verbose: int = 0):
        """Optimize the thetas parameters

        Args:
            - num_steps: number of iterations
            - verbose (int, optional): 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per 10 steps. Verbose 1 is good for timing training time, verbose 2 if you want to log loss values to a file. Please install package tdqm if you want to use verbose 1. 

        """
        self.num_steps = num_steps
        if len(self.thetas) == 0:
            if (len(self.u.parameters)) > 0:
                self.thetas = np.ones(len(self.u.parameters))
            else:
                self.thetas = np.ones(len(self.vdagger.parameters))
        self.is_trained = True
        self.thetass, self.loss_values = qtm.base.fit(
            self.u, self.vdagger, self.thetas, self.num_steps, self.loss_func, self.optimizer, verbose, is_return_all_thetas=True, **self.kwargs)
        if (len(self.u.parameters)) > 0:
            self.traces, self.fidelities, self.ce = qtm.utilities.calculate_QSP_metrics(
                self.u, self.vdagger, self.thetass, **self.kwargs)
        else:
            self.traces, self.fidelities, self.ce = qtm.utilities.calculate_QST_metrics(
                self.u, self.vdagger, self.thetass, **self.kwargs)

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
        y2 = self.fidelities
        y3 = self.traces
        fig, ax = plt.subplots()
        ax.set_xlim(int(-self.num_steps*0.05), int(self.num_steps*1.05))
        ax.set_ylim(-0.05, 1.05)
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

        animator = animation.FuncAnimation(fig, update,
                                    interval=interval, repeat=False)
        animator.save(file_name)
        return
    def save(self, ansatz, state, file_name):
        if (len(self.u.parameters)) > 0:
            qspobj = qtm.qsp.QuantumStatePreparation.load_from_compiler(
                self,
                ansatz=ansatz)
            qspobj.save(state, file_name)
        else:
            qstobj = qtm.qst.QuantumStateTomography.load_from_compiler(
                self,
                ansatz=ansatz)
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
