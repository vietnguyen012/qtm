import types, typing
from qtm.evolution import ecircuit, selection
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
    def __init__(self, params: typing.Union[typing.List, str],
                 fitness_func: types.FunctionType = None,
                 crossover_func: types.FunctionType = None,
                 mutate_func: types.FunctionType = None,
                 selection_func: types.FunctionType = None,
                 pool = None, save_progress = False) -> None:
        """_summary_

        Args:
            params (typing.Union[typing.List, str]): Other params for GA proces
            fitness_func (types.FunctionType, optional): Defaults to None.
            crossover_func (types.FunctionType, optional): Defaults to None.
            mutate_func (types.FunctionType, optional): Defaults to None.
            selection_func (types.FunctionType, optional): Defaults to None.
            pool (_type_, optional): Pool gate. Defaults to None.
            save_progress (bool, optional): is save progress or not. Defaults to False.
        """
        self.params = params
        self.fitness_func = fitness_func
        self.crossover_func = crossover_func
        self.mutate_func = mutate_func
        self.selection_func = selection_func
        self.pool = pool
        self.save_progress = save_progress
        self.best_candidate = None
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

    def load_from_file(self, file_name: str):
        file = open(file_name, 'rb')
        data = pickle.load(file)
        self.__init__(
            data.params,
            data.fitness_func,
            data.crossover_func,
            data.mutate_func,
            data.selection_func,
            data.pool, data.save_progress)
        file.close()
        return
    
    def evol(self, verbose: int = 1):
        if verbose == 1:
            bar = qtm.progress_bar.ProgressBar(
                max_value=self.num_generation, disable=False)
        for generation in range(self.num_generation):
            print(generation)
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
            if self.save_progress:
                self.populations.append(self.population)
            # Mutate
            for circuit in self.population:
                if random.random() < self.prob_mutate:
                    self.mutate_func(circuit, self.pool)

            best_score = np.min(self.scores_in_loop)
            best_index = np.argmin(self.scores_in_loop)
            if self.best_candidate.fitness > self.population[best_index].fitness:
                self.best_candidate = self.population[best_index]
            self.best_score_progress.append(best_score)
            if verbose == 1:
                bar.update(1)
            if verbose == 2 and generation % 5 == 0:
                print("Step " + str(generation) + ": " + str(best_score))
            if self.threshold(best_score):
                break
        print('End best score, end evol progress, percent target: %.1f' % best_score)
        return

    def initialize_population(self):
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
        plt.plot(self.best_score_progress)
        plt.xlabel('No. generation')
        plt.ylabel('Best score')
        plt.show()
        
    def save(self, file_name):
        file = open(file_name, 'wb')
        pickle.dump(self, file)
        file.close()
        return

    