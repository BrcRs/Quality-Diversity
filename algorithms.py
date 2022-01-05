import random
from container import hash_ind

def clone_ind(ind, icls, scls):
    indcopy = icls(i for i in ind) # Individual
    indcopy.strategy = scls(i for i in ind.strategy) # Strategy
    assert ind is not indcopy, "copy " + str(indcopy) + "\nis the same ref as\n" + str(ind)
    assert ind == indcopy, "Copy is not identical in value"
    return indcopy

# Taken from deap library
def varOr(population, toolbox, parents, lambda_, cxpb, mutpb, *clone_args):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover, mutation **or** reproduction). The modified individuals have
    their fitness invalidated. The individuals are cloned so returned
    population is independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :parents: dictionary keeping record of parents of generated individuals.
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

    This variation is named *Or* because an offspring will never result from
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
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = list(map(toolbox.clone, random.sample(population, 2)))

            # ancestor = toolbox.clone(ind1) # BUG toolbox.clone doesn't make a new instance? Why?
            ancestor = clone_ind(ind1, *clone_args) # BUG toolbox.clone doesn't make a new instance? Why?


            ### TEST REMOVE ME: BUG
            ancestor.curiosity = 666.0
            assert ind.curiosity != 666.0
            ### END TEST

            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            assert ancestor is not ind1
            offspring.append(ind1)
            parents[hash_ind(ind1)] = ancestor
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))

            ancestor = toolbox.clone(ind)
            # ancestor = clone_ind(ind, *clone_args) # BUG toolbox.clone doesn't make a new instance? Why?

            ind, = toolbox.mutate(ind)
            # assert ancestor != ind # Doesn't pass... is it even a problem? 
            # No because '==' checks the values and 'is' checks the identities
            assert ancestor is not ind

            ### TEST REMOVE ME: The test passes!
            ancestor.curiosity = 666.0
            assert ind.curiosity != 666.0
            ### END TEST

            del ind.fitness.values
            offspring.append(ind)
            parents[hash_ind(ind)] = ancestor
        else:                           # Apply reproduction
            offspring.append(random.choice(population))

    return offspring
