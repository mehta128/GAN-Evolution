#!/usr/bin/env python3

import argparse
import numpy as np
import random
import time
import csv

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import copy
from gan_class import GAN, GANDescriptor,Discriminator,Generator
import os
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from numpy.random import randint
import itertools
from metrics.fid import fid_score
from metrics import generative_score
from util.ConvLayer import Conv2dSame
from genetic_evolution import gaMuPlusLambda

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



#####################################################################################################

class MyContainer(object):
    # This class does not require the fitness attribute
    # it will be  added later by the creator
    def __init__(self, GAN):
        # Some initialisation with received values
        self.GAN = GAN


# ####################################################################################################

def eval_gan_fid(individual):
    global All_Evals
    global best_val
    global Eval_Records
    global test1
    global test2


    start_time = time.time()
    my_gan = individual.GAN
    # Train the GAN network before evaluating
    my_gan.train_gan()
    elapsed_time = time.time() - start_time
    fid_score = my_gan.getFIDScore()
    torch.cuda.empty_cache()

    # if not offspring save the fid score and time
    if my_gan.offspring == 0:
        my_gan.results[my_gan.gen_no] = [fid_score,elapsed_time]

    All_Evals = All_Evals+1

    print("Eval:",  All_Evals, " Fitness:",  fid_score, " Time:", elapsed_time)
    if fid_score < best_val:
        best_val = fid_score
        torch.save(individual.GAN.Gen_network, "bestG.pkl")
        torch.save(individual.GAN.Disc_network, "bestD.pkl")
    # Todo store data for geneting results 1. Graph for fid score and 2. Time for pareto set with gan individual
        #MAKE A GRAPH OF FID SCORE OVER GENERATIONS

        # fig = plt.figure(figsize=(5,5))
        # ax = fig.add_subplot(111)
        # ax.set_xlabel('$f(x_1)$', fontsize=25)
        # ax.set_ylabel('$f(x_2)$', fontsize=25)
        # plt.plot(test1, test2, 'b.')
        # plt.subplots_adjust(hspace=0, wspace=0, left=0.2, right=1, bottom=0.16, top=1)
        # plt.xticks(np.arange(0, np.max(nf1), 0.5), fontsize=20)
        # plt.yticks(np.arange(0, np.max(nf1), 0.5), fontsize=20)
        # plt.plot(nf1, nf2, 'r.')
        # plt.text(0, 0, str(igd_val)+' -- '+str(len(tf1)), size=15)
        # #plt.show()
        # fig.savefig("DeapGAN_"+str(my_gan_descriptor.fmeasure)+"_"+Function+'_eval_'+str(All_Evals)+'.pdf')
        # plt.close()

        # my_gan_descriptor.print_components()

    return fid_score, elapsed_time

#####################################################################################################
def inc_individual():
    global indi_no
    indi_no += 1


def init_individual(ind_class):
    # Root directory for dataset

    image_size = 64
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Grayscale(1),
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                             shuffle=True, num_workers=4)

    input_channel = 1
    output_dim = 64
    lrate = 0.001
    lossfunction = randint(0,5)

    my_gan_descriptor = GANDescriptor(input_channel, output_dim, lossfunction, lrate, dataloader, epochs,indi_no)
    inc_individual()
    g_layer = np.random.randint(5,max_layer) # Number of layers
    g_activations = [randint(0, 3) for x in range(g_layer - 1)]


    g_opChannels = gchannels[g_layer]

    g_weight_init = 0
    g_loop = 1
    d_layer = np.random.randint(5,max_layer) # Number of layers
    d_weight_init = 0
    d_activations = [randint(0, 3) for x in range(d_layer - 1)]

    d_opChannels = dchannels[d_layer]
    d_loop = 1

    my_gan_descriptor.gan_generator_initialization(g_layer, input_channel, output_dim, g_opChannels,
                                                   g_weight_init, g_activations, g_loop)
    my_gan_descriptor.gan_discriminator_initialization(d_layer, input_channel, output_dim, d_opChannels,
                                                       d_weight_init, d_activations, d_loop)
    my_gan = GAN(my_gan_descriptor)
    ind = ind_class(my_gan)
    return ind

#####################################################################################################


def cx_gan(ind1, ind2):
    """Crossover between two GANs
       The networks of the two GANs are exchanged
    """

    off1 = copy.deepcopy(ind1.GAN)

    off1.Disc_network = copy.deepcopy(ind2.GAN.Disc_network)

    ind2.GAN.Gen_network = copy.deepcopy(ind1.GAN.Gen_network)

    ind1.GAN = off1

    return ind1, ind2

#####################################################################################################


def mut_gan(individual):
    """
    Different types of mutations for the GAN.
    Only of the networks is mutated (Discriminator or Generator)
    Each time a network is mutated, only one mutation operator is applied
    """

    gan = individual.GAN
    my_gan_descriptor = gan.descriptor
    updateNetwork = False

    if np.random.uniform(0,1) < 0.6:                   # Discriminator will be mutated

        aux_network = my_gan_descriptor.Disc_network
        update_net= 'D'
        aux_network_model = gan.Disc_network
    else:                                       # Generator network will be mutated
        aux_network = my_gan_descriptor.Gen_network
        update_net = 'G'
        aux_network_model = gan.Gen_network

    # Decide the type of the mutation
    if aux_network.number_layers >=5 and aux_network.number_layers < max_layer-2:
        type_mutation = mutation_types[np.random.randint(len(mutation_types))]
    else:
        type_mutation = mutation_types[np.random.randint(2, len(mutation_types))]

     # Todo add D-G loop mutations
    # if type_mutation == "network_loops":       # The number of loops for the network learning is changed
    #     aux_network.number_loop_train = np.random.randint(nloops)+1
    # elif type_mutation == "weigt_init":             # We change weight initialization function in all layers
    #     init_w_function = init_functions[np.random.randint(len(init_functions))]
    #     aux_network.change_all_weight_init_fns(init_w_function)
    # elif type_mutation == "dimension":              # We change the number of neurons in layer
    #     aux_network.change_dimensions_in_random_layer(max_layer_size)
    # elif type_mutation == "latent":             # We change the divergence measure used by the GAN
    #     latent_distribution = lat_functions[np.random.randint(len(lat_functions))]
    #     my_gan_descriptor.latent_distribution_function = latent_distribution
    #
    # elif type_mutation == "lrate":
    #     my_gan_descriptor.lrate = np.random.choice(lrates)
    if type_mutation == "add_layer":             # We add one layer
        aux_network.number_layers += 1
        if aux_network.number_layers < 10:
            # Index where the new layer is added to maintain the convolution operation
            pos = {6: 1, 7: 3, 8: 5, 9: 7, 10: 2}
            if(update_net == 'D'):
                aux_network.list_ouput_channels = dchannels[aux_network.number_layers]

                add_from = {5: 2, 6: 8, 7: 14, 8: 20, 9: 5}

                aux_network.list_ouput_channels = dchannels[aux_network.number_layers]
                channel = dchannels[aux_network.number_layers][pos[aux_network.number_layers]]
                list_of_layers = list(aux_network_model.main.children())
                new_layer = [Conv2dSame(channel,channel, (4, 4), 1),nn.BatchNorm2d(channel)]
                activation_val = randint(0, 3)
                if activation_val == 0:
                    new_layer += [nn.LeakyReLU(0.2)]
                elif activation_val == 1:
                    new_layer += [nn.ReLU(inplace=True)]
                elif activation_val == 2:
                    new_layer += [nn.ELU(inplace=True)]

                list_of_layers = list_of_layers[:add_from[(aux_network.number_layers - 1)]] + new_layer + list_of_layers[add_from[(aux_network.number_layers - 1)]:]
                aux_network_model.main = nn.Sequential(*list_of_layers)
                aux_network.list_act_functions.insert(pos[aux_network.number_layers], activation_val)


            if(update_net == 'G'):
                # Index where the new layer is added to maintain the convolution operation
                pos = {6:1,7:3,8:5,9:7,10:2}
                add_from = {5: 3, 6: 9, 7: 15, 8: 21, 9: 6}

                aux_network.list_ouput_channels = gchannels[aux_network.number_layers]
                list_of_layers = list(aux_network_model.main.children())
                channel = gchannels[aux_network.number_layers][pos[aux_network.number_layers]]

                new_layer = [nn.ConvTranspose2d(channel, channel, 3, 1, 1, bias=False),
                             nn.BatchNorm2d(channel)]
                activation_val = randint(0,3)
                if activation_val == 0:
                    new_layer += [nn.LeakyReLU(0.2)]
                elif activation_val == 1:
                    new_layer += [nn.ReLU(inplace=True)]
                elif activation_val == 2:
                    new_layer += [nn.ELU(inplace=True)]

                list_of_layers = list_of_layers[:add_from[(aux_network.number_layers-1)]] + new_layer + list_of_layers[add_from[(aux_network.number_layers-1)]:]
                aux_network_model.main = nn.Sequential(*list_of_layers)

                aux_network.list_act_functions.insert(pos[aux_network.number_layers],activation_val)

    elif type_mutation == "del_layer":              # We remove one layer
        aux_network.number_layers -= 1
        if(aux_network.number_layers >= 5):
            #only delete the layer if the layer count is 6

            activation_pop = {5: 1, 6: 3, 7: 5, 8: 7, 9: 2}
            if(update_net == 'G'):
                # Because of conv operations, strides and kernel are fix so the removed index is also static
                remove_from ={5:3,6:9,7:15,8:21,9:6}
                aux_network.list_ouput_channels = gchannels[aux_network.number_layers]


            if(update_net == 'D'):
                remove_from = {5: 2, 6: 8, 7: 14, 8: 20, 9: 5}
                aux_network.list_ouput_channels = dchannels[aux_network.number_layers]


            # Delete the layers from the model
            from_index = remove_from[aux_network.number_layers]
            to_index = remove_from[aux_network.number_layers] + 3
            list_of_layers = list(aux_network_model.main.children())
            del list_of_layers[from_index:to_index]
            aux_network_model.main = nn.Sequential(*list_of_layers)
            # Remove the activation function from the array
            aux_network.list_act_functions.pop(activation_pop[aux_network.number_layers])


    elif type_mutation == "activation":             #

        # We change the activation function in layer
        layer_pos = randint(aux_network.number_layers-2)
        aux_network.list_act_functions[layer_pos] = randint(0, 3)
        # Calculate the index to be replaced for the selected layer
        if (update_net == 'G'):
            c = 2
            for i in range(layer_pos):
                c += 3
        elif(update_net == 'D'):
            c = 1
            for i in range(layer_pos):
                c += 3
        if aux_network.list_act_functions[layer_pos] == 0:
            aux_network_model.main[c] = nn.LeakyReLU(0.2)
        elif aux_network.list_act_functions[layer_pos] == 1:
            aux_network_model.main[c] = nn.ReLU(inplace=True)
        elif aux_network.list_act_functions[layer_pos] == 2:
            aux_network_model.main[c]  = nn.ELU(inplace=True)



    elif type_mutation == "lossfunction":             # We change the divergence measure used by the GAN
        fmeasure =randint(0, 5)
        aux_network.lossfunction = fmeasure

    return individual,


# ####################################################################################################

def init_ga():
    """
                         Definition of GA operators
    """
    creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))

    creator.create("Individual", MyContainer, fitness=creator.Fitness)

    # Structure initializers

    toolbox = base.Toolbox()
    toolbox.register("individual", init_individual, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_gan_fid)
    toolbox.register("mate", cx_gan)
    toolbox.register("mutate", mut_gan)

    if SEL == 0:
        toolbox.register("select", tools.selBest)
    elif SEL == 1:
        toolbox.register("select", tools.selTournament, tournsize=tournsel_size)
    elif SEL == 2:
        toolbox.register("select", tools.selNSGA2)

    return toolbox


#####################################################################################################

def aplly_ga_gan(toolbox, pop_size=10, gen_number=50, cxpb=0.7, mutpb=0.3):
    """
          Application of the Genetic Algorithm
    """
    hall_of_fame = 3
    offspring_count = pop_size
    pop = toolbox.population(n=pop_size)
    global hall_of
    hall_of = tools.HallOfFame(hall_of_fame)
    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    result, log_book = gaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=offspring_count, cxpb=cxpb, mutpb=mutpb,
                                                 stats=stats, halloffame=hall_of, ngen=gen_number, verbose=1)

    return result, log_book, hall_of

#####################################################################################################

def printNetwork(obj):
    network = ["layers :"+str(obj.number_layers)+" \n"]
    network.append("op channels :"+str(obj.list_ouput_channels)+" \n")
    network.append("loops :"+str(obj.number_loop_train)+" \n")
    network.append("init functions"+str(obj.init_functions)+" \n")
    network.append("activation functions :"+str(obj.list_act_functions)+" \n")
    network.append("loss function :"+str(obj.lossfunction)+ " \n")


    return  network
if __name__ == "__main__":
    #   main()
    import warnings
    warnings.filterwarnings("ignore")

    global All_Evals
    global best_val
    global Eval_Records
    global test1
    global test2
    global gchannels
    global max_layer
    global epochs
    global dchannels
    global dataroot

    indi_no = 1
    gchannels = {5: [512, 256, 128, 64, 64],
                 6: [512, 512, 256, 128, 64, 64],
                 7: [512, 512, 256, 256, 128, 64, 64],
                 8: [512, 512, 256, 256, 128, 128, 64, 64],
                 9: [512, 512, 256, 256, 128, 128, 64, 64, 64],
                 10: [512, 512, 512, 256, 256, 128, 128, 64, 64, 64]}


    dchannels = {5: [64, 128, 256, 512, 512],
                 6: [64, 64, 128, 256, 512, 512],
                 7: [64, 64, 128, 128, 256, 512, 512],
                 8: [64, 64, 128, 128, 256, 256, 512, 512],
                 9: [64, 64, 128, 128, 256, 256, 512, 512, 512],
                 10: [64, 64, 64, 128, 128, 256, 256, 512, 512, 512]}

    dataroot = "mnist_png/training"
    epochs = 1 # 20 iter per epochs- 64*20 = 1028 samples
    max_layer = 8
    ngen = 50                # Number of generations
    npop = 7
    SEL = 0                 # Selection method
    CXp = 0.2          # Crossover probability (Mutation is 1-CXp)
    Mxp = 1-CXp


    All_Evals = 0
    best_val = 10000
    Eval_Records = None
           # SelecteWWd population size
    tournsel_size = 4

    # Mutation types
    mutation_types = ["add_layer", "del_layer", "activation", "lossfunction"]
    # GA initialization
    toolb = init_ga()
    # Runs the GA
    res, logbook, hof = aplly_ga_gan(toolb, pop_size=npop, gen_number=ngen, cxpb=CXp, mutpb=Mxp)

    # Save the FID and time score to the csv file
    for ind in res:
        fname = 'C:/Users/RACHIT/Desktop/GAGAN/results/ind_'+str(ind.GAN.indi_no)+'.csv'
        file = open(fname, "w", newline='')
        w = csv.writer(file)
        w.writerow(["Generation", "Fid", "Time"])
        for key, val in ind.GAN.results.items():
            w.writerow([key, val[0], val[1]])
        file.close()
    # Save the network architecture to the text file
    for ind in res:
        fname ='C:/Users/RACHIT/Desktop/GAGAN/results/ind_'+str(ind.GAN.indi_no)+'.txt'
        file1 = open(fname, "w")
        file1.write("Fitness :"+str(ind.fitness)+" \n")
        file1.writelines(printNetwork(ind.GAN.descriptor.Disc_network))
        file1.writelines(printNetwork(ind.GAN.descriptor.Gen_network))
        file1.close()


    print("############# LOG BOOK ################")
    print(logbook)
