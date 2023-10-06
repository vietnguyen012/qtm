import types, typing
from qtm.evolution import ecircuit, selection
from datetime import datetime
import qtm.random_circuit
import qtm.state
import qtm.qcompilation
import qtm.ansatz
import random
import numpy as np
import matplotlib.pyplot as plt
import qtm.progress_bar
import pickle

class EEnvironment():

    def __init__(self, file_name: str):
        file = open(file_name, 'rb')
        data = pickle.load(file)
        self.__init__(
            data.params,
            data.fitness_func,
            data.crossover_func,
            data.mutate_func,
            data.selection_func,
            data.pool, 
            data.file_name)
        file.close()
        return
    def __init__(self, params: typing.Union[typing.Dict, str],
                 fitness_func: types.FunctionType = None,
                 crossover_func: types.FunctionType = None,
                 mutate_func: types.FunctionType = None,
                 selection_func: types.FunctionType = None,
                 pool = None, 
                 file_name: str = '') -> None:
        """_summary_

        Args:
            params (typing.Union[typing.List, str]): Other params for GA proces
            fitness_func (types.FunctionType, optional): Defaults to None.
            crossover_func (types.FunctionType, optional): Defaults to None.
            mutate_func (types.FunctionType, optional): Defaults to None.
            selection_func (types.FunctionType, optional): Defaults to None.
            pool (_type_, optional): Pool gate. Defaults to None.
            file_name (str, optional): Path of saved file.
        """
        if isinstance(params, str):
            file = open(params, 'rb')
            data = pickle.load(file)
            params = data.params
            self.params = data.params
            self.fitness_func = data.fitness_func
            self.crossover_func = data.crossover_func
            self.mutate_func = data.mutate_func
            self.selection_func = data.selection_func
            self.pool = data.pool
            self.file_name = data.file_name
            self.best_candidate = data.best_candidate
            self.current_generation = data.current_generation
            self.population = data.population
            self.populations = data.populations
            self.best_score_progress = data.best_score_progress
            self.scores_in_loop = data.scores_in_loop
        else:
            self.params = params
            self.fitness_func = fitness_func
            self.crossover_func = crossover_func
            self.mutate_func = mutate_func
            self.selection_func = selection_func
            self.pool = pool
            self.file_name = file_name
            self.best_candidate = None
            self.current_generation = 0
            self.population = []
            self.populations = []
            self.best_score_progress = []
            self.scores_in_loop = []
        self.depth = params['depth']
        self.num_circuit = params['num_circuit']  # Must mod 8 = 0
        self.num_generation = params['num_generation']
        self.num_qubits = params['num_qubits']
        self.prob_mutate = params['prob_mutate']
        self.threshold = params['threshold']
        return

    def evol(self, verbose: int = 1):
        # Pre-procssing
        if verbose == 1:
            bar = qtm.progress_bar.ProgressBar(
                max_value=self.num_generation, disable=False)
        
            
        if self.current_generation == 0:
            print("Initialize population ...")
            self.init()
            print("Start evol progress ...")
        elif self.current_generation == self.num_generation:
            return
        else:
            print(f"Continute evol progress at generation {self.current_generation} ...")
        for generation in range(self.current_generation, self.num_generation):
            print(f"Evol at generation {generation}")
            self.current_generation += 1
            self.scores_in_loop = []
            new_population = []
            # Selection
            self.population = self.selection_func(self.population)
            for i in range(0, self.num_circuit, 2):
                # Crossover
                offspring1, offspring2 = self.crossover_func(
                    self.population[i], self.population[i+1])
                new_population.extend([offspring1, offspring2])
                self.scores_in_loop.extend([offspring1.fitness, offspring2.fitness])
            self.population = new_population
            self.populations.append(self.population)
            # Mutate
            for circuit in self.population:
                if random.random() < self.prob_mutate:
                    self.mutate_func(circuit, self.pool)

            # Post-process
            best_score = np.min(self.scores_in_loop)
            best_index = np.argmin(self.scores_in_loop)
            if self.best_candidate.fitness > self.population[best_index].fitness:
                self.best_candidate = self.population[best_index]
            self.best_score_progress.append(best_score)
            self.save(self.file_name + f'ga_{self.fitness_func.__name__}_{datetime.now().strftime("%Y-%m-%d")}.envobj')
            if verbose == 1:
                bar.update(1)
            if verbose == 2 and generation % 5 == 0:
                print("Step " + str(generation) + ": " + str(best_score))
            if self.threshold(best_score):
                break
        print('End evol progress, best score ever: %.1f' % best_score)
        return

    def init(self):
        self.population = []
        num_sastify_circuit = 0
        while(num_sastify_circuit <= self.num_circuit):
            random_circuit = qtm.random_circuit.generate_with_pool(
                self.num_qubits, self.depth, self.pool)
            
            if selection.sastify_circuit(random_circuit):
                num_sastify_circuit += 1
                circuit = ecircuit.ECircuit(
                    random_circuit,
                    self.fitness_func)
                circuit.compile()
                self.population.append(circuit)
        self.best_candidate = self.population[0]
        return

    def plot(self):
        plt.plot(list(range(1, self.num_generation + 1)), self.best_score_progress)
        plt.xlabel('No. generation')
        plt.ylabel('Best score')
        plt.show()
        return 
    def save(self, file_name):
        file = open(file_name, 'wb')
        pickle.dump(self, file)
        file.close()
        return

    