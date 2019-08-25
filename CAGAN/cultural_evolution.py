from deap import algorithms
from tqdm import tqdm
import random
import numpy as np


def init_beliefspace():
    """

    Sitatuational knowledge: We have best individual found at every generation to be in our situational knowledge
    It is used to do crossover to generate new offsprings and Situational knowledge is used to do step size for mutational adjustement for some gene


    Domain knowledge is not reinitalized at every generation but contains an archive of best solution since evolution started
    1 We have DCGAN and RSGAN as our domain knowledege
    2 We augment the images and save it in our domain knowledge

    Normative knowledge:
    The normative knowledge is used for both search direction and step size
    """
    belief_space = {}

    domain = {
        "loss": 3,
        "gactivation": 1,
        "dactivation": 0,
        "dchannels": {5: [64, 128, 256, 512, 512],
                      6: [64, 64, 128, 256, 512, 512],
                      7: [64, 64, 128, 128, 256, 512, 512],
                      8: [64, 64, 128, 128, 256, 256, 512, 512],
                      9: [64, 64, 128, 128, 256, 256, 512, 512, 512],
                      10: [64, 64, 64, 128, 128, 256, 256, 512, 512, 512]},
        "gchannels": {5: [512, 256, 128, 64, 64],
                      6: [512, 512, 256, 128, 64, 64],
                      7: [512, 512, 256, 256, 128, 64, 64],
                      8: [512, 512, 256, 256, 128, 128, 64, 64],
                      9: [512, 512, 256, 256, 128, 128, 64, 64, 64],
                      10: [512, 512, 512, 256, 256, 128, 128, 64, 64, 64]}

    }
    normative = {}
    # min,max,lowerbound,upperbound,size
    normative['glayer'] = [-1,-1,10000,10000]
    normative['dlayer'] = [-1,-1,10000,10000]
    normative['gactivation'] = [-1,-1,10000,10000]
    normative['dactivation'] = [-1,-1,10000,10000]
    normative['loss'] = [-1,-1,10000,10000]

    belief_space['domain'] = domain
    belief_space['normative'] = normative
    return belief_space


def updateNormativeKnowledge(normative_knw,elites):
    minelite = elites[0]
    fitness = minelite.fitness.values[0]
    # f(x(t)) < L(t)
    if (fitness < normative_knw['glayer'][2]):
        normative_knw['glayer'][0] = minelite.GAN.descriptor.Gen_network.number_layers
        normative_knw['glayer'][2] = fitness
    # update dlayer
    # f(x(t)) < L(t)
    if (fitness < normative_knw['dlayer'][2]):
        normative_knw['dlayer'][0] = minelite.GAN.descriptor.Disc_network.number_layers
        normative_knw['dlayer'][2] = fitness
        # update loss
        # f(x(t)) < L(t)
    if (fitness < normative_knw['loss'][2]):
        normative_knw['loss'][0] = minelite.GAN.descriptor.lossfunction
        normative_knw['loss'][2] = fitness
        # update gactivations
        # f(x(t)) < L(t)
    if (fitness < normative_knw['gactivation'][2]):
        normative_knw['gactivation'][0] = minelite.GAN.descriptor.Gen_network.list_act_functions
        normative_knw['gactivation'][2] = fitness
        # update dactivations
        # f(x(t)) < L(t)
    if (fitness < normative_knw['dactivation'][2]):
        normative_knw['dactivation'][0] = minelite.GAN.descriptor.Disc_network.list_act_functions
        normative_knw['dactivation'][2] = fitness
    # Set upperbound
    maxelite = elites[2]
    fitness = maxelite.fitness.values[0]
        # f(x(t)) < U(t)
    if (fitness < normative_knw['glayer'][3]):
       normative_knw['glayer'][1] = maxelite.GAN.descriptor.Gen_network.number_layers
       normative_knw['glayer'][3] = fitness
        # f(x(t)) < U(t)
    if (fitness < normative_knw['dlayer'][3]):
           normative_knw['dlayer'][1] =  maxelite.GAN.descriptor.Disc_network.number_layers
           normative_knw['dlayer'][3] = fitness
    # f(x(t)) < U(t)
    if (fitness < normative_knw['loss'][3]):
        normative_knw['loss'][1] =  maxelite.GAN.descriptor.lossfunction
        normative_knw['loss'][3] = fitness
    # f(x(t)) < U(t)
    if (fitness < normative_knw['gactivation'][3]):
        normative_knw['gactivation'][1] = maxelite.GAN.descriptor.Gen_network.list_act_functions
        normative_knw['gactivation'][3] =fitness
    # f(x(t)) < U(t)
    if(fitness < normative_knw['dactivation'][3]):
        normative_knw['dactivation'][1] = maxelite.GAN.descriptor.Disc_network.list_act_functions
        normative_knw['dactivation'][3] = fitness

def gaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):

    logbook = algorithms.tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    #  Evaluate the fitness before starting
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # BELIEF INIT
    belief_space = init_beliefspace()
    # Sort the population and take top 3 elites
    population.sort(key=lambda ind: ind.fitness)
    elites = population[:3]
    updateNormativeKnowledge(belief_space["normative"],elites)

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in tqdm(range(1, ngen + 1)):

        print("_________________GA GEN: "+str(gen)+"______________________")

        # Invalidate fitness of whole pop to train and evaluate
        for ind in population:
            ind.GAN.gen_no = gen
            ind.GAN.offspring = 0
            del ind.fitness.values
        # Train and Evaluate the individuals
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit


        # Update normative knowledge using elite individuals
        # set min and max for G,D - layers,set activations array, set output channels array and loss function
        # Use normative knowledge to guide the mutational adjustment
        population.sort(key=lambda ind: ind.fitness)
        elites = population[:3]
        updateNormativeKnowledge(belief_space["normative"], elites)

        # Vary the population
        offspring = varOr(population, toolbox, lambda_, halloffame,belief_space,cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = []
        for ind in offspring:
            if not ind.fitness.valid:
                ind.GAN.offspring = 1
                invalid_ind.append(ind)


        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)




        # Select the next generation population
        population[:] = toolbox.select(population +  offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

def varOr(population, toolbox, lambda_,halloffame,belief_space, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover, mutation **or** reproduction). The modified individuals have
    their fitness invalidated. The individuals are cloned so returned
    population is independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param lambda\_: The number of children to produce
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: The final population.

    The variation goes as follow. On each of the *lambda_* iteration, it
    selects one of the three operations; crossover, mutation or reproduction.
    In the case of a crossover, two individuals are selected at random from
    the parental population :math:`P_\mathrm{p}`, those individuals are cloned
    using the :meth:`toolbox.clone` method and then mated using the
    :meth:`toolbox.mate` method. Only the first child is appended to the
    offspring population :math:`P_\mathrm{o}`, the second child is discarded.
    In the case of a mutation, one individual is selected at random from
    :math:`P_\mathrm{p}`, it is cloned and then mutated using using the
    :meth:`toolbox.mutate` method. The resulting mutant is appended to
    :math:`P_\mathrm{o}`. In the case of a reproduction, one individual is
    selected at random from :math:`P_\mathrm{p}`, cloned and appended to
    :math:`P_\mathrm{o}`.

    This variation is named *Or* beceause an offspring will never result from
    both operations crossover and mutation. The sum of both probabilities
    shall be in :math:`[0, 1]`, the reproduction probability is
    1 - *cxpb* - *mutpb*.
    """
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:
            # Apply crossover
            ind1,ind2= list(map(toolbox.clone, random.sample(population, 2)))

            # Add crossover with hall of fame - situational influence
            if np.random.uniform(0,1) < 0.6:
                ind2 = toolbox.clone(halloffame.items[0])
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind,belief_space)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))

    return offspring
