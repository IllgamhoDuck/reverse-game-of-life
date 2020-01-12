# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    print.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: hypark <marvin@42.fr>                      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/01/11 20:59:36 by hypark            #+#    #+#              #
#    Updated: 2020/01/11 22:19:42 by hypark           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

def print_train_information(args, model):
    print("----------MODEL INFORMATION--------")
    print("%-20s %d" % ("kernel number", model.kernel_num))
    print("%-20s {}\n".format(model.kernel_size_list) % "kernel list")

    print("----------HYPERPARAMETER-----------")
    print("%-20s %d" % ("population", args.population_num))
    print("%-20s %d" % ("generation", args.generation_num))
    print("%-20s %.2f" % ("survive ratio", args.survive_ratio))
    print("%-20s %.2f" % ("mutation_probability", args.mutation_probability))
    print("%-20s %.2f" % ("mutation_area", args.mutation_area))
    print("%-20s %.2f\n" % ("threshold", args.threshold))
    
    print("----------TRAIN INFORMATION--------")
    print("%-20s %d" % ("dataset size", args.dataset_limit))
    print("%-20s %d" % ("batch size", args.batch_size))
    print("%-20s %d" % ("print every", args.print_every))
    print("%-20s %d\n" % ("save every", args.save_every))

    print("----------LOAD INFORMATION---------")
    if args.load_model:
        print("%-20s %s" % ("model load name", args.model_name_load))
        print("%-20s %s\n" % ("kernel save name", args.kernel_name_load))
    else:
        print("NOTHING IS LOADED\n")

    print("----------SAVE INFORMATION--------")
    print("%-20s %s" % ("model save name", args.model_name_save))
    print("%-20s %s\n" % ("kernel save name", args.kernel_name_save))

def print_predict_information(args, model, path):
    print("----------MODEL INFORMATION--------")
    print("%-20s %d" % ("kernel number", model.kernel_num))
    print("%-20s {}\n".format(model.kernel_size_list) % "kernel list")
    
    print("--------PREDICT INFORMATION--------")
    print("%-20s %d\n" % ("dataset size", 50000))

    print("----------LOAD INFORMATION---------")
    print("%-20s %s" % ("model load name", args.model_name_load))
    print("%-20s %s\n" % ("kernel save name", args.kernel_name_load))

    print("-------SUBMISSION INFORMATION------")
    print("Submission file will be generated at -> ", end="")
    print("%s\n" % path)
