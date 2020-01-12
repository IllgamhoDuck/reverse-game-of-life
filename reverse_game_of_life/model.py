# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    model.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: hypark <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/11 16:58:57 by hypark            #+#    #+#              #
#    Updated: 2020/01/11 21:41:37 by hypark           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from copy import deepcopy
import os
import sys
import pickle
import math
import random
import numpy as np

class Kernel(object):
    """One kernel"""
    def __init__(self, kernel_size):
        """
        Prepare neccesary values

        IMAGE
        [1 2]
        [3 4]

        KERNEL SIZE 
        PAD
        pad_height = kernel_height - 1
        pad_height = kernel_width - 1

        Example - kernel size (2 x 2)
        [1 2 0]
        [3 4 0]
        [0 0 0]

        :param Tuple kernel_size: (h, w) kernel height and width 
        """
        self.kernel_size = kernel_size
        self.kernel_h = kernel_size[0]
        self.kernel_w = kernel_size[1]
        self.kernel = None

        # create the kernel
        self._create_kernel()
    
    def __call__(self, x):
        return self._forward(x)

    def __str__(self):
        return "kernel\nshape : {}\n{}\n".format(self.kernel.shape,
                                                 self.kernel)
 
    def __repr__(self):
        return "kernel({})".format(self.kernel_size)
    
    def _create_kernel(self):
        self._kernel_initialize()

    def _kernel_initialize(self):
        self.kernel = np.random.randn(self.kernel_h,
                                      self.kernel_w)
 
    def _pad(self, x, h, w):
        padded_x = np.zeros((h + self.kernel_h - 1,
                            w + self.kernel_w - 1))
        padded_x[:h,:w] = x
        return padded_x
    
    def _convolve(self, x, h, w):
        convolved_x = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                convolved_x[i][j] = (self.kernel * x[i:i+self.kernel_h,
                                                    j:j+self.kernel_w]).sum()
        return convolved_x       

    def _forward(self, x):
        h, w = x.shape
        x = self._pad(x, h, w)
        x = self._convolve(x, h, w)
        return x


class MultiKernels(object):
    """Multi kernels"""
    def __init__(self, kernel_num, threshold, kernel_size_list=None):
        """
        Prepare neccesary values

        :param Tuple kernel_size_list: [(h, w), ... , (h, w)]
            -> list of kernel size tuples height and width
            -> if None fill the list to default value (3, 3)
        """
        self.kernel_num = kernel_num
        self._threshold = threshold
        self.kernel_size_list = kernel_size_list
        self.kernel_list = []

        # Initialize kernels size
        if (kernel_size_list):
            if (len(kernel_size_list) == kernel_num):
                self.kernel_size_list = kernel_size_list
            else:
                raise IndexError
        else:
            self.kernel_size_list = [(3, 3) for _ in range(self.kernel_num)]

        # Initialize kernels
        self._create_kernels()

    def __call__(self, x):
        return self._forward(x)

    def __str__(self):
        status = ""
        for i in range(1, self.kernel_num + 1):
            status += "kernel {}\n".format(i)
            status += str(self.kernel_list[i - 1])
            status += "\n"
        return status
    
    def __repr__(self):
        return "MultiKernels({}, {}, {})".format(self.kernel_num,
                                                self._threshold,
                                                self.kernel_size_list)
        
    def change_threshold(self, threshold):
        print("Threshold changed from {} ".format(self._threshold), end="")
        self._threshold = threshold
        print("to {}".format(self._threshold))
        

    def _create_kernels(self):
        for kernel_size in self.kernel_size_list:
            self.kernel_list.append(Kernel(kernel_size))
    
    def _create_board(self, x):
        x_shape = x.shape
        x = x.reshape(-1)
        x[np.where((x >= self._threshold).reshape(-1))] = 1
        x[np.where((x < self._threshold).reshape(-1))] = 0
        return x.reshape(x_shape)
    
    def _forward(self, x):
        result = np.zeros_like(x, dtype=float)
        for kernel in self.kernel_list:
            result += kernel(x)
        
        # Apply threshold
        return self._create_board(result)

    def save_model(self, path):
        model = [self.kernel_num,
                 self._threshold,
                 self.kernel_size_list]
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print("Model saved to {} successfully!".format(path))

    def save_kernels(self, path):
        kernels = []
        for kernel in self.kernel_list:
            kernels.append(kernel.kernel)
        with open(path, 'wb') as f:
            pickle.dump(kernels, f)
        print("Kernel saved to {} successfully!".format(path))

    def load_kernels(self, path):
        with open(path, 'rb') as f:
            kernels = pickle.load(f)
        for kernel_, kernel in zip(self.kernel_list, kernels):
            kernel_.kernel = kernel 
        print("Kernel loaded from {} successfully!".format(path))


class Genetic(object):
    """Genetic Algorithm to train the model"""
    def __init__(self,
                 population_num,
                 kernel_num,
                 threshold=0.5,
                 kernel_size_list=None):
        """
        Choose the model

        :param int population_num: Choose the number of population
        :param int kernel_num: Choose the size of kernel of one model
        :param float threshold: Choose the threshold that model will use to generate the board
        :param Tuple kernel_size_list: [(h, w), ... , (h, w)]
            -> list of kernel size tuples height and width
            -> if None fill the list to default value (3, 3)
        """
        self.population_num = population_num
        self.kernel_num = kernel_num
        self.threshold = threshold
        self.kernel_size_list = kernel_size_list
        self.model_info = [self.kernel_num,
                           self.threshold,
                           self.kernel_size_list]
        self.dataset_num = None

        # Create the Models to populate
        self.population()

    def save_model(self, path):
        """Saving the best model"""
        self.model_list[0].save_model(path)

    def save_kernel(self, path):
        """Saving the best kernel"""
        self.model_list[0].save_kernels(path)

    def load_model(self, path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        self.model_info = model
        self.kernel_num = self.model_info[0]
        self.threshold = self.model_info[1]
        self.kernel_size_list = self.model_info[2]
        self.population()
        print("Model was loaded from {} successfully!".format(path))

    def load_kernel(self, path):
        self.model_list[0].load_kernels(path)

    def __str__(self):
        status = ""
        for i, model in enumerate(self.model_list):
            status += "model {} ".format(i + 1)
            status += repr(model)
            status += "\n"
        return status

    def __repr__(self):
        return "Genetic({}, {}, {}, {})".format(self.population_num,
                                                *self.model_info)

    def __call__(self,
                 dataset,
                 batch_size=None,
                 generation_num=500,
                 survive_ratio=0.5,
                 mutation_probability=0.5,
                 mutation_area=0.2,
                 verbose=True,
                 print_every=10,
                 save_every=100,
                 model_path=None,
                 kernel_path=None):
        """
        :param dataset: Training dataset [id / step / start board / end board]
        :param batch_size: How much dataset will gonna use for each fitness score
        :param generation_num: How much generation we are going to run?
        :param survive_ratio: At the selection step how much model will survive?
        :param mutation_probability: Choose the probability to mutate model
        :param mutation area: When we mutate the model how many area we will gonna mutate?
        :param s
        """
        self.dataset = dataset
        self.dataset_len = len(self.dataset)
        if batch_size:
            self.batch_size = batch_size
        else:
            self.batch_size = self.dataset_len
        # Used to know how far it used dataset
        self.current_index = 0
        self.generation_num = generation_num
        self.survive_ratio = survive_ratio
        self.mutation_probability = mutation_probability
        self.mutation_area = mutation_area
        for generation in range(1, generation_num + 1):
            self.evolve()
            if verbose and (generation % print_every == 0):
                self.log(generation)
            if (generation % save_every == 0):
                if (model_path and kernel_path):
                    self.save_model(model_path)
                    self.save_kernel(kernel_path)

            # Update the current index to search next dataset in the next generation
            self.current_index = (self.current_index + self.batch_size) % self.dataset_len

    def best_model_score(self):
        """Using the best model predict the past"""
        # Use the best model
        model = self.model_list[0]

        # Show the model score
        print("Score will be tested in 50,000")
        model_score = self.model_score(model, mode='test')
        print("The model we are going to use has score : {}".format(model_score))

    def predict(self, predict_dataset, predict_dataset_index=None, visual=None):
        # Use the best model
        model = self.model_list[0]

        # Predict the result
        if (predict_dataset_index == None):
            print("Currently showing prediction for every dataset is not supported")
            print("Please choose specific index of dataset.")
            print("predict(dataset, predict_dataset_index=0)")
            raise NotImplementedError

        data = predict_dataset[predict_dataset_index]
        x = data['end']
        for _ in range(data['step']):
            x = model(x)

        if visual:
            # Show the result
            print("This is the end board")
            visualize_board(data['end'])
            print("This is the answer")
            visualize_board(data['start'])
            print("This is the prediction")
            visualize_board(x)
        else:
            return x

    def log(self, generation):
        """Print out the current situation of training"""
        self.score_list = []
        # Calculate the score
        for model in self.model_list:
            self.score_list.append(self.model_score(model, mode='train'))
        print("Generation [ {} / {} ]".format(generation, self.generation_num))
        print("Population {} - {}".format(self.population_num, self.score_list))

    def change_threshold(self, threshold):
        """Change the threshold of all the models"""
        for model in self.model_list:
            model.change_threshold(threshold)

    def change_population(self, population_num):
        """Change the population number and populate models again"""
        self.population_num = population_num
        self.population()
        print("Changed to {} population and repopulate model".format(self.population_num))

    def change_model(self, kernel_num, threshold=.5, kernel_size_list=None):
        """Change the model specification and populate models again"""
        self.kernel_num = kernel_num
        self.threshold = threshold
        self.kernel_size_list = kernel_size_list
        self.model_info = [self.kernel_num,
                           self.threshold,
                           self.kernel_size_list]
        self.population()
        print("Changed the model spec to {} and repopulate model".format(repr(self.model_list[0])))

    def population(self):
        self.model_list = []
        for _ in range(self.population_num):
            self.model_list.append(MultiKernels(*self.model_info))

        # Update the kernel size list to default value if it is not designated
        if self.kernel_size_list is None:
            self.kernel_size_list = self.model_list[0].kernel_size_list
            self.model_info = [self.kernel_num,
                               self.threshold,
                               self.kernel_size_list]

    def evolve(self):
        self.fitness()
        self.selection()
        self.crossover()
        self.mutation()

    def fitness(self):
        self.score_list = []
        # Calculate the score
        for model in self.model_list:
            self.score_list.append(self.model_score(model, mode='train'))

        # Sort the model list
        self.model_list = [self.model_list[index] for index in np.argsort(self.score_list)[::-1]]

    def model_score(self, model, mode='train'):
        total_score = 0
        total_count = 0
        if mode == 'train':
            for i in range(self.batch_size):
                index = (self.current_index + i) % self.dataset_len
                data = self.dataset[index + 1]
                x = data['end']
                for _ in range(data['step']):
                    x = model(x)
                total_score += self.score_method(x, data['start'])
                total_count += 1
        elif mode == 'test':
            for _, data in self.dataset.items():
                x = data['end']
                for _ in range(data['step']):
                    x = model(x)
                total_score += self.score_method(x, data['start'])
                total_count += 1
        else:
            raise NotImplementedError
        return round(total_score / total_count, 4)

    def score_method(self, output, target):
        return sum((output == target).reshape(-1))/400

    def selection(self):
        survive_num = round(self.population_num * self.survive_ratio)
        self.model_list = [self.model_list[index] for index in range(survive_num)]

    def crossover(self):
        # Reproduce childs to fill the population
        while (self.population_num != len(self.model_list)):
            parent = random.sample(self.model_list, 2)

            # Make child
            son, daugter = self.make_child(parent)
            self.model_list.append(son)
            if (self.population_num != len(self.model_list)):
                self.model_list.append(daugter)

    def make_child(self, parent):
        # parent
        dad = parent[0]
        mom = parent[1]

        # make child
        son = MultiKernels(*self.model_info)
        daughter = MultiKernels(*self.model_info)

        # Give the child a parent power
        for i, (dad_kernel, mom_kernel) in enumerate(zip(dad.kernel_list, mom.kernel_list)):
            # kernel reproduce
            reproduced = self.reproduce(dad_kernel.kernel, mom_kernel.kernel)
            son.kernel_list[i].kernel = reproduced[0]
            daughter.kernel_list[i].kernel = reproduced[1]

        return son, daughter

    def reproduce(self, dad, mom):
        overlap = np.random.binomial(1, 0.5, size=dad.shape).astype('bool')
        son = deepcopy(dad)
        daugter = deepcopy(mom)
        son[overlap] = mom[overlap]
        daugter[overlap] = dad[overlap]
        return son, daugter

    def mutation(self):
        for i, model in enumerate(self.model_list):
            # Don't mutate the 1st, 2nd model
            if (i == 0 or i == 1):
                continue
            if random.random() < self.mutation_probability:
                self.mutate_model(model)

    def mutate_model(self, model):
        for kernel in model.kernel_list:
            k_shape = kernel.kernel.shape
            self.mutate_kernel(kernel, k_shape)

    def mutate_kernel(self, kernel, k_s):
        kernel_mutate_area = np.random.binomial(1, self.mutation_area, size=k_s)
        kernel_random = np.random.randn(*k_s)
        kernel.kernel += kernel_mutate_area
        kernel.kernel[kernel_mutate_area] = kernel_random[kernel_mutate_area]

