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
from tunable_gan_class_optimization import xavier_init, GAN, GANDescriptor
from Class_F_Functions import pareto_frontier, FFunctions, igd
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# This function set the seeds of the tensorflow function
# to make this notebook's output stable across runs


def reset_graph(seed=42):
    global All_Evals
    tf.reset_default_graph()
    tf.set_random_seed(All_Evals)
    np.random.seed(All_Evals)
    random.seed(All_Evals)


#####################################################################################################

class MyContainer(object):
    # This class does not require the fitness attribute
    # it will be  added later by the creator
    def __init__(self, thegan_descriptor):
        # Some initialisation with received values
        self.gan_descriptor = thegan_descriptor


# ####################################################################################################

def eval_gan_approx_igd(individual):
    global All_Evals
    global best_val
    global Eval_Records
    global test1
    global test2

    reset_graph(myseed)

    start_time = time.time()
    my_gan_descriptor = individual.gan_descriptor
    my_gan = GAN(my_gan_descriptor)
    my_gan.training_definition()
    final_samples = my_gan.separated_running("fixed", ps_all_x, mb_size, number_epochs,  number_samples, print_cycle)
    elapsed_time = time.time() - start_time

    # We reset the network since we do not need to keep the results of the training
    # Also due to problems with deepcopy for tensorflow

    nf1, nf2 = MOP_f.evaluate_mop_function(final_samples)
    tf1, tf2, _ = pareto_frontier(nf1, nf2)

    igd_val = igd((np.vstack((tf1, tf2)).transpose(), np.vstack((test1, test2)).transpose()))[0]
    All_Evals = All_Evals+1

    gan_code = my_gan_descriptor.codify_components(nlayers, init_functions, act_functions, divergence_measures, lat_functions)
    print("Eval:",  All_Evals, " PS_Size:", len(tf1), " Fitness:",  igd_val, " Time:", elapsed_time)
    if All_Evals == 1:
        Eval_Records = np.array(gan_code+[All_Evals, len(tf1), igd_val, elapsed_time])
    else:
        Eval_Records = np.vstack((Eval_Records, np.array(gan_code+[All_Evals, len(tf1), igd_val, elapsed_time])))

    if igd_val < best_val:
        best_val = igd_val

        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        ax.set_xlabel('$f(x_1)$', fontsize=25)
        ax.set_ylabel('$f(x_2)$', fontsize=25)
        plt.plot(test1, test2, 'b.')
        plt.subplots_adjust(hspace=0, wspace=0, left=0.2, right=1, bottom=0.16, top=1)
        plt.xticks(np.arange(0, np.max(nf1), 0.5), fontsize=20)
        plt.yticks(np.arange(0, np.max(nf1), 0.5), fontsize=20)
        plt.plot(nf1, nf2, 'r.')
        plt.text(0, 0, str(igd_val)+' -- '+str(len(tf1)), size=15)
        #plt.show()
        fig.savefig("DeapGAN_"+str(my_gan_descriptor.fmeasure)+"_"+Function+'_eval_'+str(All_Evals)+'.pdf')
        plt.close()

        # my_gan_descriptor.print_components()

    return igd_val, elapsed_time

#####################################################################################################


def init_individual(ind_class):

    generator_n_hidden = np.random.randint(nlayers)+1                                 # Number of hidden layers
    discriminator_n_hidden = np.random.randint(nlayers)+1                             # Number of hidden layers

    generator_dim_list = [np.random.randint(max_layer_size)+1 for _ in range(generator_n_hidden)]
    discriminator_dim_list = [np.random.randint(max_layer_size)+1 for _ in range(discriminator_n_hidden)]

    gen_number_loop_train = np.random.randint(nloops)+1                            # Number loops for training generator
    disc_number_loop_train = np.random.randint(nloops)+1                        # Number loops for training discriminator

    # divergence_measures = ["Standard_Divergence","Total_Variation","Forward_KL","Reverse_KL","Pearson_Chi_squared","Squared_Hellinger","Least_squared"]

    fmeasure = divergence_measures[np.random.randint(len(divergence_measures))]

    i_g_function = init_functions[np.random.randint(len(init_functions))]   # List or random init functions for generator
    i_d_function = init_functions[np.random.randint(len(init_functions))]   # List or random init functions for discriminator

    generator_init_functions = []
    discriminator_init_functions = []

    generator_act_functions = []
    discriminator_act_functions = []

    for i in range(generator_n_hidden+1):
        generator_init_functions.append(i_g_function)
        if i == generator_n_hidden:
            generator_act_functions.append(None)        # Activation functions for all layers in generator
        else:
            generator_act_functions.append(act_functions[np.random.randint(len(act_functions))])

    for i in range(discriminator_n_hidden+1):
        discriminator_init_functions.append(i_d_function)
        if i == discriminator_n_hidden:
            discriminator_act_functions.append(None)        # Activation functions for all layers in discriminator
        else:
            discriminator_act_functions.append(act_functions[np.random.randint(len(act_functions))])

    latent_distribution_function = lat_functions[np.random.randint(len(lat_functions))]             # Distribution for latent random samples

    lrate = np.random.choice(lrates)

    my_gan_descriptor = GANDescriptor(X_dim, z_dim, latent_distribution_function, fmeasure, lrate)

    my_gan_descriptor.gan_discriminator_initialization(generator_n_hidden, generator_dim_list,
                                                       generator_init_functions,  generator_act_functions, gen_number_loop_train)
    my_gan_descriptor.gan_generator_initialization(discriminator_n_hidden, discriminator_dim_list,
                                                   discriminator_init_functions,  discriminator_act_functions, disc_number_loop_train)

    ind = ind_class(my_gan_descriptor)

    return ind

#####################################################################################################


def cx_gan(ind1, ind2):
    """Crossover between two GANs
       The networks of the two GANs are exchanged
    """
    off1 = copy.deepcopy(ind1)
    off1.gan_descriptor.Disc_network = copy.deepcopy(ind2.gan_descriptor.Disc_network)
    ind2.gan_descriptor.Gen_network = copy.deepcopy(ind1.gan_descriptor.Gen_network)

    ind1 = off1

    return ind1, ind2

#####################################################################################################


def mut_gan(individual):
    """
    Different types of mutations for the GAN.
    Only of the networks is mutated (Discriminator or Generator)
    Each time a network is mutated, only one mutation operator is applied
    """

    my_gan_descriptor = individual.gan_descriptor
    if random.random() < 0.5:                   # Discriminator will be mutated
        aux_network = my_gan_descriptor.Disc_network
    else:                                       # Generator network will be mutated
        aux_network = my_gan_descriptor.Gen_network

    if aux_network.number_hidden_layers < nlayers:
        type_mutation = mutation_types[np.random.randint(len(mutation_types))]

    else:
        type_mutation = mutation_types[np.random.randint(1, len(mutation_types))]

    if type_mutation == "network_loops":       # The number of loops for the network learning is changed
        aux_network.number_loop_train = np.random.randint(nloops)+1

    elif type_mutation == "add_layer":             # We add one layer
        layer_pos = np.random.randint(aux_network.number_hidden_layers)+1
        lay_dims = np.random.randint(max_layer_size)+1
        init_w_function = init_functions[np.random.randint(len(init_functions))]
        init_a_function = act_functions[np.random.randint(len(act_functions))]
        aux_network.network_add_layer(layer_pos, lay_dims, init_w_function, init_a_function)

    elif type_mutation == "del_layer":              # We remove one layer
        aux_network.network_remove_random_layer()

    elif type_mutation == "weigt_init":             # We change weight initialization function in all layers
        init_w_function = init_functions[np.random.randint(len(init_functions))]
        aux_network.change_all_weight_init_fns(init_w_function)

    elif type_mutation == "activation":             # We change the activation function in layer
        layer_pos = np.random.randint(aux_network.number_hidden_layers)
        init_a_function = act_functions[np.random.randint(len(act_functions))]
        aux_network.change_activation_fn_in_layer(layer_pos, init_a_function)

    elif type_mutation == "dimension":              # We change the number of neurons in layer
        aux_network.change_dimensions_in_random_layer(max_layer_size)

    elif type_mutation == "divergence":             # We change the divergence measure used by the GAN
        fmeasure = divergence_measures[np.random.randint(len(divergence_measures))]
        my_gan_descriptor.fmeasure = fmeasure

    elif type_mutation == "latent":             # We change the divergence measure used by the GAN
        latent_distribution = lat_functions[np.random.randint(len(lat_functions))]
        my_gan_descriptor.latent_distribution_function = latent_distribution

    elif type_mutation == "lrate":
        my_gan_descriptor.lrate = np.random.choice(lrates)

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

    toolbox.register("evaluate", eval_gan_approx_igd)
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


if __name__ == "__main__":  # Example python3 GAN_Descriptor_Deap.py 0 1000 10 1 30 10 5 50 10 2 0 10 0

    #   main()
    parser = argparse.ArgumentParser()
    parser.add_argument('integers', metavar='int', type=int, choices=range(3000),
                        nargs='+', help='an integer in the range 0..3000')
    parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum,
                        default=max, help='sum the integers (default: find the max)')

    # The parameters of the program are set or read from command line

    global All_Evals
    global best_val
    global Eval_Records
    global test1
    global test2

    All_Evals = 0
    best_val = 10000
    Eval_Records = None

    args = parser.parse_args()
    myseed = args.integers[0]               # Seed: Used to set different outcomes of the stochastic program
    number_samples = args.integers[1]       # Number samples to generate by the GAN (number_samples=1000)
    n = args.integers[2]                    # Number of Variables of the MO function (n=10)
    Function = "F"+str(args.integers[3])    # MO function to optimize, all in F1...F9, except F6
    z_dim = args.integers[4]                # Dimension of the latent variable  (z_dim=30)
    nlayers = args.integers[5]              # Maximum number of layers for generator and discriminator (nlayers = 10)
    nloops = args.integers[6]               # Maximum number of loops for training a network  (nloops = 5)
    max_layer_size = args.integers[7]       # Maximum size of the layers  (max_layer_size = 50)
    npop = args.integers[8]                 # Population size
    ngen = args.integers[9]                 # Number of generations
    SEL = args.integers[10]                 # Selection method
    CXp = args.integers[11]*0.01            # Crossover probability (Mutation is 1-CXp)
    nselpop = args.integers[12]             # Selected population size
    if len(args.integers) > 13:
        tournsel_size = args.integers[13]   # Tournament value
    else:
        tournsel_size = 4

    All_Evals = 0                                     # Tracks the number of evaluations
    best_val = 10000.0                                # Tracks best value among solutions

    k = 1000                                          # Number of samples of the Pareto set for computing approximation (k=1000)
    MOP_f = FFunctions(n, Function)                   # Creates a class containing details on MOP
    ps_all_x = MOP_f.generate_ps_samples(k)           # Generates k points from the Pareto Set
    test = MOP_f.generate_ps_samples(k)
    pf1, pf2 = MOP_f.evaluate_mop_function(ps_all_x)  # Evaluate the points from the Pareto Set
    test1, test2 = MOP_f.evaluate_mop_function(ps_all_x)
    X_dim = n                                         # Number of variables to approximate

    # List of activation functions the networks can use
    act_functions = [None, tf.nn.relu, tf.nn.elu, tf.nn.softplus,
                     tf.nn.softsign, tf.sigmoid, tf.nn.tanh]

    # List of weight initialization functions the networks can use
    init_functions = [xavier_init, tf.random_uniform, tf.random_normal]

    #  List of latent distributions VAE can use
    lat_functions = [np.random.uniform, np.random.normal]

    #  List of divergence measures
    divergence_measures = ["Total_Variation", "Standard_Divergence", "Forward_KL", "Reverse_KL", "Pearson_Chi_squared", "Least_squared"]
    lrates = [0.00001]#[0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    # Mutation types
    mutation_types = ["add_layer", "network_loops", "del_layer", "weigt_init", "activation", "dimension", "divergence", "latent", "lrate"]

    mb_size = 150           # Minibatch size
    number_epochs = 1001    # Number epochs for training
    print_cycle = 1001      # Frequency information is printed

    # GA initialization
    toolb = init_ga()

    # Runs the GA
    res, logbook, hof = aplly_ga_gan(toolb, pop_size=npop, gen_number=ngen, cxpb=CXp, mutpb=1-CXp)

    # Save the configurations of all networks evaluated
    fname = "GAN_Evals_"+str(myseed)+"_"+Function+"_N_"+str(npop)+"_ngen_"+str(ngen)+"_Sel_"+str(SEL)+"_X_" + str(X_dim)
    np.save(fname, Eval_Records)

    # Examples of how to call the function
    # ./GAN_Descriptor_Deap.py 111 1000 10 1 30 10 5 50 20 1000 0 20 10 5
