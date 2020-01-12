# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    util.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: hypark <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/11 15:42:44 by hypark            #+#    #+#              #
#    Updated: 2020/01/11 20:37:26 by hypark           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import matplotlib.pyplot as plt
import seaborn as sns

# visualize board - visualize the conway board using seaborn heatmap
# check valid game - Check the conway game model works exactly same as original conway game
# preprocess_kernel_list = change the kernel list from string to tuple

def visualize_board(board):
    plt.figure(figsize=(7, 7))
    sns.heatmap(board,
                square=True,
                linewidths=0.1,
                linecolor='gray',)
                # xticklabels=False,
                # yticklabels=False)

def check_valid_game(dataset, conway_game):
    total_len = len(dataset)
    correct = 0
    for i in tqdm(range(1, total_len + 1)):
        result = conway_game(dataset[i]["start"], dataset[i]["step"])
        if (result == dataset[i]["end"]).all():
            correct += 1
        sys.stdout.write("\rChecking the game is valid... {} / {}".format(correct, total_len))
        sys.stdout.flush()

def preprocess_kernel_list(kernel_list):
    new_kernel_list = []
    for kernel in kernel_list:
        new_kernel_list.append((int(kernel[0]), int(kernel[2])))
    return new_kernel_list
