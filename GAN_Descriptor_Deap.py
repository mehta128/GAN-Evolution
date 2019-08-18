#!/usr/bin/env python3

import argparse
import tensorflow as tf
import numpy as np
import random
import time

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
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from numpy.random import randint
import itertools
from metrics.fid import fid_score
from metrics import generative_score



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
    my_gan.train_gan()
    elapsed_time = time.time() - start_time

    # We reset the network since we do not need to keep the results of the training
    # Also due to problems with deepcopy for tensorflow
    fid_score = my_gan.getFIDScore()
    All_Evals = All_Evals+1

    print("Eval:",  All_Evals, " Fitness:",  fid_score, " Time:", elapsed_time)
    torch.cuda.empty_cache()
    if fid_score < best_val:
        best_val = fid_score

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


def init_individual(ind_class):
    # Root directory for dataset
    dataroot = "mnist_png/training"
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
                                             shuffle=True, num_workers=2)

    input_channel = 1
    output_dim = 64
    lrate = 0.0002
    lossfunction = randint(0,5)
    epochs = 2
    my_gan_descriptor = GANDescriptor(input_channel, output_dim, lossfunction, lrate, dataloader, epochs)

    g_layer = np.random.randint(5,8) # Number of layers
    g_activations = [randint(0, 3) for x in range(g_layer - 1)]


    g_opChannels = gchannels[g_layer]

    g_weight_init = 0
    g_loop = 1
    d_layer = np.random.randint(5,8) # Number of layers
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
    off1 = copy.deepcopy(ind1)
    off1.Disc_network = copy.deepcopy(ind2.Disc_network)
    ind2.Gen_network = copy.deepcopy(ind1.Gen_network)

    ind1 = off1

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


    if random.random() < 0.5:                   # Discriminator will be mutated
        aux_network = my_gan_descriptor.Disc_network
        update_net= 'D'
    else:                                       # Generator network will be mutated
        aux_network = my_gan_descriptor.Gen_network
        update_net = 'G'

    # Decide the type of the mutation
    if aux_network.number_layers < 9:
        type_mutation = mutation_types[np.random.randint(len(mutation_types))]
    else:
        type_mutation = mutation_types[np.random.randint(1, len(mutation_types))]

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
        aux_network.list_act_functions.append(randint(0, 3))
        if(update_net == 'D'):
            aux_network.list_ouput_channels = dchannels[aux_network.number_layers]
        if(update_net == 'G'):
            aux_network.list_ouput_channels = gchannels[aux_network.number_layers]
    elif type_mutation == "del_layer":              # We remove one layer
        aux_network.number_layers -= 1
        aux_network.list_act_functions = aux_network.list_act_functions[:aux_network.number_layers]
        if(update_net == 'D'):
            aux_network.list_ouput_channels = dchannels[aux_network.number_layers]
        if(update_net == 'G'):
            aux_network.list_ouput_channels = gchannels[aux_network.number_layers]

    elif type_mutation == "activation":             # We change the activation function in layer
        layer_pos = np.random.randint(aux_network.number_layers)
        aux_network.list_act_functions[layer_pos] = randint(0, 3)
    elif type_mutation == "lossfunction":             # We change the divergence measure used by the GAN
        fmeasure =randint(0, 5)
        aux_network.lossfunction = fmeasure


    if(update_net == 'D'):
        del gan.Disc_network
        gan.init_d_model = True
        gan.Disc_network = Discriminator(aux_network).to(gan.device)
    if(update_net == 'G'):
        del gan.Gen_network
        gan.init_g_model = True
        gan.Gen_network = Generator(aux_network).to(gan.device)




    return individual,


# ####################################################################################################

def init_ga():
    """
                         Definition of GA operators
    """

    # Minimization of the IGD measure

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
    # elif SEL == 1:
    #     toolbox.register("select", tools.selTournament, tournsize=tournsel_size)
    # elif SEL == 2:
    #     toolbox.register("select", tools.selNSGA2)

    return toolbox


#####################################################################################################

def aplly_ga_gan(toolbox, pop_size=10, gen_number=50, cxpb=0.7, mutpb=0.3):
    """
          Application of the Genetic Algorithm
    """

    pop = toolbox.population(n=pop_size)
    hall_of = tools.HallOfFame(pop_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    result, log_book = algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size, cxpb=cxpb, mutpb=mutpb,
                                                 stats=stats, halloffame=hall_of, ngen=gen_number, verbose=1)

    return result, log_book, hall_of

#####################################################################################################


if __name__ == "__main__":
    #   main()
    global All_Evals
    global best_val
    global Eval_Records
    global test1
    global test2
    global gchannels
    gchannels = {5: [512, 256, 128, 64, 64],
                 6: [512, 512, 256, 128, 64, 64],
                 7: [512, 512, 256, 256, 128, 64, 64],
                 8: [512, 512, 256, 256, 128, 128, 64, 64],
                 9: [512, 512, 256, 256, 128, 128, 64, 64, 64],
                 10: [512, 512, 512, 256, 256, 128, 128, 64, 64, 64]}

    global  dchannels
    dchannels = {5: [64, 128, 256, 512, 512],
                 6: [64, 64, 128, 256, 512, 512],
                 7: [64, 64, 128, 128, 256, 512, 512],
                 8: [64, 64, 128, 128, 256, 256, 512, 512],
                 9: [64, 64, 128, 128, 256, 256, 512, 512, 512],
                 10: [64, 64, 64, 128, 128, 256, 256, 512, 512, 512]}

    All_Evals = 0
    best_val = 10000
    Eval_Records = None

    ngen = 3                # Number of generations
    npop = 2
    SEL = 0                 # Selection method
    CXp = 0.03            # Crossover probability (Mutation is 1-CXp)
    # nselpop = args.integers[12]             # SelecteWWd population size
    # if len(args.integers) > 13:
    #     tournsel_size = args.integers[13]   # Tournament value
    # else:
    #     tournsel_size = 4

    All_Evals = 0                                     # Tracks the number of evaluations
    best_val = 10000.0                                # Tracks best value among solutions

    # Mutation types
    mutation_types = ["add_layer", "del_layer", "activation", "lossfunction"]
    # GA initialization
    toolb = init_ga()

    # Runs the GA
    res, logbook, hof = aplly_ga_gan(toolb, pop_size=npop, gen_number=ngen, cxpb=CXp, mutpb=1-CXp)

    # Examples of how to call the function
    # ./GAN_Descriptor_Deap.py 111 1000 10 1 30 10 5 50 20 1000 0 20 10 5
