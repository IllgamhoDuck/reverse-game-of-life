# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: hypark <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/11 14:56:52 by hypark            #+#    #+#              #
#    Updated: 2020/01/11 23:07:52 by hypark           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

#!/usr/bin/env python
# -*- coding: utf-8 -*- 
#####################################################################
##                     Setting Environment                         ##
#####################################################################

if __package__ == None:
    __package__ = "main"

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# Reverse game of life library
# util.py - print / conway game of life
# dataset.py - read csv file and convert to pd / preprocess dataset
import reverse_game_of_life as rgol

# TODO : Choose the path where to save model and kernel
model_path = './model'
kernel_path = './save'

# TODO : Choose the path for train set and test set
dataset_path = './resources/'

# TODO : Choose the path for submission.csv file
submission_path = './'

def dir_checker():
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    if not os.path.isdir(kernel_path):
        os.mkdir(kernel_path)

#####################################################################
##                        ARGUMENT PARSER                          ##
#####################################################################

def argument_parser():
    parser = argparse.ArgumentParser(
    """
    Choose to train or predict and set the arguments.
    """
    )

    # Train mode
    parser.add_argument('-t', '--train', action='store_true', help="Train mode")

    ## Init the model
    parser.add_argument('--population_num', type=int, default=30, help="Population number")
    parser.add_argument('--threshold', type=float, default=0.5,
            help="Threshold decide the final board's pixel is 0 or 1 based on the sum of all the filtered board")
    parser.add_argument('--kernel_num', type=int, default=3, help="Filter(Kernel) number")
    parser.add_argument('--kernel_list', type=tuple, nargs='*',
            default=[('3', ',', '3'), ('4', ',', '4'), ('5', ',', '5')], help="Filter(Kernel) number")

    ## Train the model
    parser.add_argument('--dataset_limit', type=int, default=50, help="Choosing the dataset size to use for training")
    parser.add_argument('--batch_size', type=int, default=50, help="How many dataset to use for one generation")

    ## Hyperparameter
    parser.add_argument('--generation_num', type=int, default=2000, help="How much generation will do for evolve?")
    parser.add_argument('--survive_ratio', type=float, default=0.5, help="How many population survive for 1 generation")
    parser.add_argument('--mutation_probability', type=float, default=0.5, help="What is the possibility to mutate the each population")
    parser.add_argument('--mutation_area', type=float, default=0.2, help="How big will the mutation area will be")

    # Log
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--print_every', type=int, default=10, help="Print the result per ? generation")
    parser.add_argument('--save_every', type=int, default=100, help="Save the result per ? generation")

    # Save
    parser.add_argument('--model_name_save', type=str, default="model.pkl", help='Model name to be saved')
    parser.add_argument('--kernel_name_save', type=str, default="kernel.pkl", help='Kernel name to be saved')

    # Load
    parser.add_argument('--load_model', action='store_true', help="Load the model, the setting for model will be replaced with the model information")
    parser.add_argument('--load_kernel', action='store_true', help="Load the kernel")
    parser.add_argument('--model_name_load', type=str, default="model.pkl", help="Use the model saved")
    parser.add_argument('--kernel_name_load', type=str, default="kernel.pkl", help="Use the kernel saved")

    # Predict mode
    parser.add_argument('-p', '--predict', action='store_true', help="Predict mode. The model and kernel will be loaded depends on the argument --model_name_load and --kernel_name_load so please choose the propriate name. Option --load_model, --load_kernel will be setted True automatically.")

    # Dataset load
    parser.add_argument('--train_set', type=str, default="train.csv", help="Name of the train file")
    parser.add_argument('--test_set', type=str, default="test.csv", help="Name of the test file")

    # Parse the information
    args = parser.parse_args()

    return args

def argument_checker(args):
    if (args.train and args.predict) or (not args.train and not args.predict):
        print("Choose train or predict model")
        raise AttributeError

def parse_argument():
    args = argument_parser()
    argument_checker(args)
    return args

#####################################################################
##                    MODEL TRAIN / PREDICT                        ##
#####################################################################

def process_model(args, dataset_path):
    # Build the dataset depend on the train / predict
    dataset = rgol.dataset.build_dataset(args, dataset_path)

    # Preprocess kernel list
    kernel_list = rgol.util.preprocess_kernel_list(args.kernel_list)

    # Initialize model
    model = rgol.model.Genetic(population_num=args.population_num,
                               kernel_num=args.kernel_num,
                               threshold=args.threshold,
                               kernel_size_list=kernel_list)

    # Train / Predict
    if args.train:
        train_model(model, dataset, args)
    else:
        predict_board(model, dataset, args)

def train_model(model, dataset, args):
    # Generate the save path
    model_save_path = os.path.join(model_path, args.model_name_save)
    kernel_save_path = os.path.join(kernel_path, args.kernel_name_save)

    # Load the model & kernel if neccessary
    if args.load_model:
        model.load_model(os.path.join(model_path, args.model_name_load))
    if args.load_kernel:
        model.load_kernel(os.path.join(kernel_path, args.kernel_name_load))

    # Information about training
    rgol.print.print_train_information(args, model)

    # Start training
    if args.verbose:
        print("TRAINING IS STARTED!")

    model(dataset=dataset,
          batch_size=args.batch_size,
          verbose=args.verbose,
          print_every=args.print_every,
          save_every=args.save_every,
          model_path=model_save_path,
          kernel_path=kernel_save_path,
          generation_num=args.generation_num,
          survive_ratio=args.survive_ratio,
          mutation_probability=args.mutation_probability,
          mutation_area=args.mutation_area)

    print("TRAINING DONE!")

def predict_board(model, dataset, args):
    # Load the model & kernel
    model.load_model(os.path.join(model_path, args.model_name_load))
    model.load_kernel(os.path.join(kernel_path, args.kernel_name_load))

    # Information about predicting
    path = os.path.join(submission_path, 'submission.csv')
    rgol.print.print_predict_information(args, model, path)

    # Generate Submission.csv file
    result = []
    for i in tqdm(range(1, len(dataset) + 1)):
        output = model.predict(dataset, i).reshape(-1)
        id_array = np.array([i])
        result.append(np.append(id_array, output))

    # change it to numpy array
    result = np.array(result, dtype=int)

    # Generate index
    columns = ['start.{}'.format(i) for i in range(1, 401)]
    columns = ['id'] + columns

    # Generate submission file
    result = pd.DataFrame(result, columns=columns)
    result.to_csv(path, index=False)


#####################################################################
##                             MAIN                                ##
#####################################################################

if __name__ == "__main__":
    # Generate the Directory for model and kernel if not there
    dir_checker()

    # Parse the argument 
    args = parse_argument()

    # Execute Model
    process_model(args, dataset_path)

