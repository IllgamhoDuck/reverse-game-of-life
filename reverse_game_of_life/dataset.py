# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    dataset.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: hypark <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/11 16:16:23 by hypark            #+#    #+#              #
#    Updated: 2020/01/11 22:00:12 by hypark           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import pandas as pd
from tqdm import tqdm

def csv_to_pd(path):
    return pd.read_csv(path)

def preprocess_dataset(pandas_dataset, type="train"):
    dataset = {}
    if (type == "train"):
        for data in tqdm(pandas_dataset.values):
            dataset[data[0]] = {"step": data[1],
                                "start": data[2:402].reshape(20, 20),
                                "end": data[402:802].reshape(20, 20)}
    elif (type == "test"):
        for data in tqdm(pandas_dataset.values):
            dataset[data[0]] = {"step": data[1],
                                "end": data[2:402].reshape(20, 20)}
    return dataset

def build_trainset(args, dataset_path):
    print("Building train set")
    train_pd = csv_to_pd(os.path.join(dataset_path, args.train_set))
    train_set = preprocess_dataset(train_pd)

    # Build a small dataset reference the dataset limit
    small_train_set = {}
    for i in range(1, args.dataset_limit + 1):
        small_train_set[i] = train_set[i]
    print("Dataset limit is {0}. Current dataset has {0} data points".format(args.dataset_limit))
    print("If you want to change the limit modify argument [--dataset_limit]\n")

    return small_train_set

def build_testset(args, dataset_path):
    print("Building test set")
    test_pd = csv_to_pd(os.path.join(dataset_path, args.test_set))
    test_set = preprocess_dataset(test_pd, type="test")
    return test_set

def build_dataset(args, dataset_path):
    if args.train:
        return build_trainset(args, dataset_path)
    else:
        return build_testset(args, dataset_path)
