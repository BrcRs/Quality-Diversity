import cma
import gym, gym_fastsim # don't remove gym_fastsim, necessary
from deap import *
import numpy as np
from fixed_structure_nn_numpy import SimpleNeuralControllerNumpy
from scipy.spatial import KDTree
import math
import warnings

import datetime

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

import algorithms as algo

import array
import random
import operator
import math
import os.path

from plot import *

from scoop import futures

#from novelty_search_vanila import *
# from novelty_search import *
from container import Archive, Grid, hash_ind
import os

import sys

# import grid_management

# import ea_dps

# Individual generator
def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size)) # Individual
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size)) # Strategy
    return ind



def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children
        return wrappper
    return decorator


class Experiment:
    registered_envs={}

    # all parameters of a run are here, change it or duplicate it to explore different possibilities
    # the parameters are grouped into subset of parameters to limite their duplication.
    # They are concatenated below to create registered_envs entries.



    fastsim_env1={
        'gym_name': 'FastsimSimpleNavigation-v0',
        'env_params': {"still_limit": 10, 'reward_kind': "continuous"},

        'nb_input': 5, # number of NN inputs
        'nb_output': 2, # number of NN outputs
        'nb_layers': 2, # number of layers
        'nb_neurons_per_layer': 10, # number of neurons per layer

        'episode_nb_step': 1000, # maximum number of steps during an episode
        'episode_reward_kind': 'final', # 2 possible values: 'final' (the reward of an episode is the last observed reward and 'cumul' (the reward of an episode is the sum of all observed rewards
        'episode_bd': 'robot_pos', # the info key value to use as a bd
        'episode_bd_slice': (0,2,None), # specify the slice of the BD you are interested in (start, stop, step), see slice function. put (None, None, None) if you want everything
        'episode_bd_kind': 'final', # only final for the moment
        'episode_log': {'collision': 'cumul', 
                        'dist_obj': 'final', 
                        'exit_reached': 'final', 
                        'robot_pos': 'final'},

        'dim_grid': [100, 100],
        'grid_min_v': [0,0],
        'grid_max_v': [600,600],
        'goal': [60,60],

        'watch_max': 'dist_obj', # watching for the max in the corresponding info entry
    }

    fastsim_env2={
        'gym_name': 'FastsimSimpleNavigation-v0',
        'env_params': {"still_limit": 10, 'reward_kind': "collisions"},

        'nb_input': 5, # number of NN inputs
        'nb_output': 2, # number of NN outputs
        'nb_layers': 2, # number of layers
        'nb_neurons_per_layer': 10, # number of neurons per layer

        'episode_nb_step': 1000, # maximum number of steps during an episode
        'episode_reward_kind': 'cumul', # 2 possible values: 'final' (the reward of an episode is the last observed reward and 'cumul' (the reward of an episode is the sum of all observed rewards
        'episode_bd': 'robot_pos', # the info key value to use as a bd
        'episode_bd_slice': (0,2,None), # specify the slice of the BD you are interested in (start, stop, step), see slice function. put (None, None, None) if you want everything
        'episode_bd_kind': 'final', # only final for the moment
        'episode_log': {'collision': 'cumul', 
                        'dist_obj': 'final', 
                        'exit_reached': 'final', 
                        'robot_pos': 'final'},

        'dim_grid': [100, 100],
        'grid_min_v': [0,0],
        'grid_max_v': [600,600],
        'goal': [60,60],
        'watch_max': 'collision', # watching for the max in the corresponding info entry

    }

    ea_generic={
        'min_value': -30, # min genotype value
        'max_value': 30, # max genotype value
        'min_strategy': 0.5, # min value for the mutation
        'max_strategy': 3, # max value for the mutation
        'nb_gen': 200, # number of generations
        'mu': 100, # population size
        'lambda': 200, # number of individuals generated # TODO isn't it supposed to be equal to mu?
        'nov_k': 15, # k parameter of novelty search
        'nov_add_strategy': "random", # archive addition strategy (either 'random' or 'novel' or 'Cully')
        'nov_lambda': 6, # number of individuals added to the archive
    }

    ea_random_sampling={
        'min_value': -30, # min genotype value
        'max_value': 30, # max genotype value
        'min_strategy': 0.5, # min value for the mutation
        'max_strategy': 3, # max value for the mutation
        'nb_gen': 0, # number of generations
        'mu': 200100, # population size
        'lambda': 200, # number of individuals generated
        'nov_k': 15, # k parameter of novelty search
        'nov_add_strategy': "random", # archive addition strategy (either 'random' or 'novel')
        'nov_lambda': 6, # number of individuals added to the archive
    }

    # possible selection
    # (noselection, random, pareto, 
    # score_based, pop_based, pop & archive based)


    # considered value between fitness, novelty, curiosity, 
    # novelty & local quality
    ea_NS={
        'quality': 'NS', # can be either NS, FIT or FIT+NS or NSLC
    }

    ea_FIT={
        'quality': 'FIT', # can be either NS, FIT or FIT+NS or NSLC
    }

    ea_FIT_NS={
        'quality': 'FIT+NS', # can be either NS, FIT or FIT+NS or NSLC
    }

    ea_NS_LC={
        'quality': 'NSLC', # can be either NS, FIT or FIT+NS or NSLC
    }
    registered_val = ['NS', 'FIT', 'FIT+NS', 'NSLC', 'curiosity']


    # Fastsim with NS
    registered_envs["fastsim_NS"]={}
    registered_envs["fastsim_NS"].update(fastsim_env2)
    registered_envs["fastsim_NS"].update(ea_generic)
    registered_envs["fastsim_NS"].update(ea_NS)


    # Fastsim with FIT
    registered_envs["fastsim_FIT"]={}
    registered_envs["fastsim_FIT"].update(fastsim_env2)
    registered_envs["fastsim_FIT"].update(ea_generic)
    registered_envs["fastsim_FIT"].update(ea_FIT)

    # Fastsim with FIT and NS (multi-objective approach)
    registered_envs["fastsim_FIT_NS"]={}
    registered_envs["fastsim_FIT_NS"].update(fastsim_env2)
    registered_envs["fastsim_FIT_NS"].update(ea_generic)
    registered_envs["fastsim_FIT_NS"].update(ea_FIT_NS)

    # Fastsim with FIT and NS (multi-objective approach)
    registered_envs["fastsim_NS_LC"]={}
    registered_envs["fastsim_NS_LC"].update(fastsim_env2)
    registered_envs["fastsim_NS_LC"].update(ea_generic)
    registered_envs["fastsim_NS_LC"].update(ea_NS_LC)

    # Fastsim with random sampling
    registered_envs["fastsim_RANDOM"]={}
    registered_envs["fastsim_RANDOM"].update(fastsim_env1)
    registered_envs["fastsim_RANDOM"].update(ea_random_sampling)
    registered_envs["fastsim_RANDOM"].update(ea_FIT_NS)

    # Fastsim with map_elites
    registered_envs["fastsim_MAP_ELITES"]={}
    registered_envs["fastsim_MAP_ELITES"].update(fastsim_env2)
    registered_envs["fastsim_MAP_ELITES"].update(ea_generic)
    registered_envs["fastsim_MAP_ELITES"].update(ea_NS)
    registered_envs["fastsim_MAP_ELITES"]['nb_gen'] = 20000
    registered_envs["fastsim_MAP_ELITES"]['lambda'] = 2
    registered_envs["fastsim_MAP_ELITES"]['nov_lambda'] = 2 

    # def __init__(self, env_name=""):
    #     # env_name="fastsim_MAP_ELITES"
    #     # change this variable to choose the environment you are interested in 
    #     # (one among the keys of registered_envs)
    #     if env_name == "":
    #         self.env_name = None
    #     else:
    #         self.env_name = env_name
    

    def __init__(self, cont=None, sel=None, val=None):
        self.custom_env = {}
        self.custom_env.update(self.ea_generic)
        self.custom_env['container'] = "undefined"
        self.custom_env['quality'] = "undefined"
        self.custom_env['selection'] = "undefined"

        self.custom_env['nov_add_strategy'] = "Cully"
        # self.custom_env['nov_add_strategy'] = "random"
        if cont != None:
            self.set_container(cont)
        if sel != None:
            self.set_selection(sel)
        if val != None:
            self.set_value(val)

    def set_add_strategy(self, strat):
        the_valid = ["random", "novel", "Cully"]
        if not strat in the_valid:
            raise ValueError("ERROR: unknown add strategy(%s), valid alternatives are", the_valid, ""%(strat))
        self.custom_env['nov_add_strategy'] = strat

    def set_container(self, cont_name):
        self.custom_env['container'] = cont_name

    def set_selection(self, sel):
        self.custom_env['selection'] = sel
        if sel == 'random':
            self.custom_env.update(self.fastsim_env1)
        else:
            self.custom_env.update(self.fastsim_env2)
    
    def set_value(self, v_name):
        if v_name not in self.registered_val:
            raise NameError("Unknown value: " + v_name + "\nPossible options: " + str(self.registered_val))
        self.custom_env['quality'] = v_name # Before the key was 'selection'

    def set_parameters(self, params):
        warnings.warn('Validity of parameters is unchecked')
        self.custom_env.update(params)

    def get_env_name(self):
        return self.custom_env['container'] + "_" + self.custom_env['selection'] + "_" + self.custom_env['quality']

    def eval_nn(self, genotype, env, resdir, render=False, dump=False, name=""):
        """ Evaluation of a neural network. Returns the fitness, the behavior descriptor and a log of what happened
            Consider using dump=True to generate log files. These files are put in the resdir directory.
        """
        nbstep=self.custom_env["episode_nb_step"]
        nn=SimpleNeuralControllerNumpy(self.custom_env["nb_input"],
                                    self.custom_env["nb_output"],
                                    self.custom_env["nb_layers"],
                                    self.custom_env["nb_neurons_per_layer"])
        nn.set_parameters(genotype)
        observation = env.reset()
        observation, reward, done, info = env.step([0]*self.custom_env["nb_output"]) # if we forget that, the initial perception may be different from one eval to another... 
        #print("First observation: "+str(observation)+" first pos: "+str(env.get_robot_pos()))
        if (dump):
            f={}
            for k in info.keys():
                fn=resdir+"/traj_"+k+"_"+name+".log"
                if (os.path.exists(fn)):
                    cpt=1
                    fn=resdir+"/traj_"+k+"_"+name+"_%d.log"%(cpt)
                    while (os.path.exists(fn)):
                        cpt+=1
                        fn=resdir+"/traj_"+k+"_"+name+"_%d.log"%(cpt)
                f[k]=open(fn,"w")

        action_scale_factor = env.action_space.high

        episode_reward=0
        episode_bd=None
        episode_log={}
        for t in range(nbstep):
            if render:
                env.render()
            action=nn.predict(observation)
            action=action_scale_factor*np.array(action)
            #print("Observation: "+str(observation)+" Action: "+str(action))
            observation, reward, done, info = env.step(action) 
            if (self.custom_env["episode_reward_kind"] == "cumul"):
                episode_reward+=reward

            for k in self.custom_env["episode_log"].keys():
                if (self.custom_env["episode_log"][k] == "cumul"):
                    if (k not in episode_log.keys()):
                        episode_log[k] = info[k]
                    else:
                        episode_log[k] += info[k]
            if(dump):
                for k in f.keys():
                    if (isinstance(info[k], list) or isinstance(info[k], tuple)):
                        data=" ".join(map(str,info[k]))
                    else:
                        data=str(info[k])
                    f[k].write(data+"\n")
            if(done):
                break
        if (dump):
            for k in f.keys():
                f[k].close()

        if (self.custom_env["episode_reward_kind"] == "final"):
            episode_reward=reward
            
        if (self.custom_env["episode_bd_kind"] == "final"):
            episode_bd=info[self.custom_env["episode_bd"]][slice(*self.custom_env["episode_bd_slice"])]
            
        for k in self.custom_env["episode_log"].keys():
            if (self.custom_env["episode_log"][k] == "final"):
                episode_log[k] = info[k]
        
        #print("End of eval, t=%d, total_dist=%f"%(t,total_dist))
        return episode_reward, episode_bd, episode_log





    def launch_ea(self, env, mu=100, lambda_=200, cxpb=0., mutpb=0.7, ngen=100, verbose=False, resdir="res", maps_elites = True):
        # I set cxpb to 0 to disable crossover


        if (self.custom_env['quality']=="FIT+NS"):
            creator.create("MyFitness", base.Fitness, weights=(-1.0,1.0))

        elif (self.custom_env['quality']=="FIT"):
            creator.create("MyFitness", base.Fitness, weights=(-1.0,)) # fitness = distance to obj, that's why we minimize it 

        elif (self.custom_env['quality']=="NS"):
            creator.create("MyFitness", base.Fitness, weights=(1.0,))

        elif (self.custom_env['quality']=="NSLC"):
            creator.create("MyFitness", base.Fitness, weights=(1.0,1.0))
        
        elif (self.custom_env['quality'] == "curiosity"):
            creator.create("MyFitness", base.Fitness, weights=(1.0,))

        else:
            print("Variante inconnue: " + self.custom_env['quality'])

        creator.create("Individual", array.array, typecode="d", fitness=creator.MyFitness, strategy=None, curiosity=0.)
        creator.create("Strategy", array.array, typecode="d")

        nn=SimpleNeuralControllerNumpy(self.custom_env["nb_input"],
                                    self.custom_env["nb_output"],
                                    self.custom_env["nb_layers"],
                                    self.custom_env["nb_neurons_per_layer"])
        center=nn.get_parameters()

        IND_SIZE=len(center)


        random.seed()

        # Preparation of the EA with the DEAP framework. See https://deap.readthedocs.io for more details.
        toolbox = base.Toolbox()
        toolbox.register("individual", generateES, creator.Individual, creator.Strategy, IND_SIZE, 
                        self.custom_env["min_value"], 
                        self.custom_env["max_value"], 
                        self.custom_env["min_strategy"], 
                        self.custom_env["max_strategy"])

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxESBlend, alpha=0.1)
        toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)

        # TODO is this where we change selection operator?
        # TODO this is kind of useless to leave this as we change it later
        toolbox.register("select", tools.selNSGA2)
        
        toolbox.register("map",futures.map)
        toolbox.decorate("mate", checkStrategy(self.custom_env["min_strategy"]))
        toolbox.decorate("mutate", checkStrategy(self.custom_env["min_strategy"]))
        toolbox.register("evaluate", self.eval_nn, env=env, resdir=resdir)


        population = toolbox.population(n=mu)
        paretofront = tools.ParetoFront()
        
        fbd=open(resdir+"/bd.log","w")
        finfo=open(resdir+"/info.log","w")
        ffit=open(resdir+"/fit.log","w")
        progress=open(resdir+"/progress.log","w")

        nb_eval=0

        ##
        ### Initial random generation: beginning
        ##
        
        # grid container
        grid = Grid()

        # archive container
        archive = None

        # parents will record the parent of who
        # {hash_ind(ind) : hash_ind(parent)}
        parents = {}
        # curio
        # {ind : change in curiosity}
        curio = {} # change in curiosity for each ind to apply to its parent

        # Evaluate the individuals with an invalid (i.e. not yet evaluated) fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses_bds = toolbox.map(toolbox.evaluate, invalid_ind)
        nb_eval += len(invalid_ind)
        finfo.write("## Generation 0 \n")
        finfo.flush()
        ffit.write("## Generation 0 \n")
        ffit.flush()
        progress.write("## Generation 0 \n")
        progress.flush()
        sum_quality=0
        max_quality = -math.inf
        sum_novelty =0
        t=0
        for ind, (fit, bd, log) in zip(invalid_ind, fitnesses_bds):
            #print("Fit: "+str(fit)) 
            #print("BD: "+str(bd))
            
            if (self.custom_env['quality']=="FIT+NS"):
                ind.fitness.values=(fit,0)
            elif (self.custom_env['quality']=="FIT"):
                ind.fitness.values=(fit,)
            elif (self.custom_env['quality']=="NS"):
                ind.fitness.values=(0,)
            elif (self.custom_env['quality']=="NSLC"):
                ind.fitness.values=(0,0)
            elif (self.custom_env['quality']=="curiosity"):
                ind.fitness.values=(0,)
            t=t+1
            ind.fit = fit # fit is the number of collisions? => YES
            ind.log = log
            ind.bd = bd
            sum_quality= sum_quality+ ind.fit
            sum_novelty= sum_novelty+ 0 # TODO why +0?
            if max_quality > fit: # we want to minimize the fitness which is at least -1000 and increases with collisions
              max_quality = fit
            fbd.write(" ".join(map(str,bd))+"\n")
            fbd.flush()
            finfo.write(str(log)+"\n")
            finfo.flush()
            ffit.write(str(fit)+"\n")
            ffit.flush()
            # The addition to the grid is based on fitness here
            # TODO Make the grid work like the archive, maybe?
            # DONE do only if container = grid
            if self.custom_env['container'] == 'grid':
                grid.add_to_grid(ind, fit, curio=curio, toolbox=toolbox,
                                        dim=self.custom_env['dim_grid'], 
                                        min_v=self.custom_env['grid_min_v'], 
                                        max_v=self.custom_env['grid_max_v'])
        progress.write("0"+" "+str(sum_quality)+" "+ str(sum_novelty) + " " + str(max_quality)+"\n")
        progress.flush()     
        if paretofront is not None:
            paretofront.update(population)

        # DONE do only if container = archive
        if self.custom_env['container'] == 'archive':
            archive = Archive.update_score(population, population, None, curio=curio, toolbox=toolbox,
                            k=self.custom_env['nov_k'],
                            add_strategy=self.custom_env['nov_add_strategy'],
                            _lambda=self.custom_env['nov_lambda'])

        ## Update the parents' curiosity
        container = {"archive" : archive, "grid" : grid}
        for ind in container[self.custom_env["container"]].pop:
            if hash_ind(ind) in parents.keys() and parents[hash_ind(ind)] in curio.keys():
                ind.curiosity = max(0, ind.curiosity + curio[parents[hash_ind(ind)]])

        for ind in population:
            if (self.custom_env['quality']=="FIT+NS"):
                ind.fitness.values=(ind.fit,ind.novelty)
            elif (self.custom_env['quality']=="FIT"):
                ind.fitness.values=(ind.fit,)
            elif (self.custom_env['quality']=="NS"):
                ind.fitness.values=(ind.novelty,)
            elif (self.custom_env['quality']=="NSLC"):
                ind.fitness.values=(ind.novelty,ind.lc)
            elif (self.custom_env['quality']=="curiosity"):
                ind.fitness.values=(ind.curiosity,)
            


            #print("Fit=%f Nov=%f"%(ind.fit, ind.novelty))

        indexmax, valuemax = max(enumerate([i.log[self.custom_env['watch_max']] for i in population]), key=operator.itemgetter(1))

        ##
        ### Initial random generation: end
        ##
        # Selected individuals for variation
        # select_pop = population.copy() # TODO Warning! (Deep) copy necessary? YES copy atleast
        select_pop = []
        # Begin the generational process
        for gen in range(1, ngen + 1):
            finfo.write("## Generation %d \n"%(gen))
            finfo.flush()
            ffit.write("## Generation %d \n"%(gen))
            ffit.flush()
            progress.write("## Generation %d \n"%(gen))
            progress.flush()
            """
            if (gen%10==0):
                print("+",end="", flush=True)
            else:
                print(".",end="", flush=True)
            """
            ## reset curio
            curio = {}
            # Variation of the previously selected population
            # DONE adapt variation
            # Note: in Cully2017, they didn't use crossover

            #### From collection
            # The collection can wether be the population or the archive or the grid
            # so collection == container?

            ## Case collection + no_sel
            # => offspring = pop because pop is already generated at random
            if self.custom_env['selection'] == 'noselection':
                offspring = population.copy()
                parents = dict() # no ancestors because generated at random
            ## Case collection + random sel
            # => offspring = polynomial variation of the filtered collection, aka container
            elif self.custom_env['selection'] in ["random", "pareto", "score", "pop", "pop&arch"]:
                # by default, cxpb = 0., no crossover
                offspring = algo.varOr(population, toolbox, parents, lambda_, cxpb, mutpb, creator.Individual, creator.Strategy)
            ## Case collection + score
            # Selection with roulette or tournament, or score-proportionate of collection

            #### From population
            ## Case collection + pop based
            # new offspring generated from current offsprings + parents
            # Then selection with roulette or tournament, or score-proportionate

            ## Case collection + Pareto
            # => offspring = variation of the filtered pop (filter with NSGA-II)

            
            ## old ways ##############
            # if self.custom_env['container'] == "grid" :
            #     # special treatment of the grid
            #     offspring = algorithms.varAnd(random.sample(list(grid.values()), 2),toolbox,cxpb,mutpb) # len(offspring) = ... 2?? Not good! TODO
            # elif self.custom_env['container'] == "archive" :
            #     offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb) # len(offspring) = lambda
            #########################

            # Evaluate the individuals with an invalid (i.e. not yet evaluated) fitness
            invalid_ind = [ind for ind in offspring]
            fitnesses_bds = toolbox.map(toolbox.evaluate, invalid_ind)
            nb_eval+=len(invalid_ind)

            for ind, (fit, bd, log) in zip(invalid_ind, fitnesses_bds):
                #print("Fit: "+str(fit)+" BD: "+str(bd)) 
                if (self.custom_env['quality']=="FIT+NS"):
                    ind.fitness.values=(fit,0)
                elif (self.custom_env['quality']=="FIT"):
                    ind.fitness.values=(fit,)
                elif (self.custom_env['quality']=="NS"):
                    ind.fitness.values=(0,)
                elif (self.custom_env['quality']=="NSLC"):
                    ind.fitness.values=(0,0)
                elif (self.custom_env['quality']=="curiosity"):
                    ind.fitness.values=(0,)
                ind.fit = fit
                ind.bd = bd
                ind.log=log
                fbd.write(" ".join(map(str,bd))+"\n")
                fbd.flush()
                finfo.write(str(log)+"\n")
                finfo.flush()
                ffit.write(str(fit)+"\n")
                ffit.flush()
                # DONE change that to generalize to all container types


            ## Choice of the collection
            collection = []
            if self.custom_env['container'] == 'grid':
                collection = grid.get_pop().copy()
            elif self.custom_env['container'] == 'archive':
                collection = archive.get_pop().copy()
            else:
                raise ValueError("Unknown container type: " + self.custom_env['container'] 
                + "\nPossible values are 'grid' and 'archive'")

            pq = []
            #### From population
            ## Case collection + pop based
            # new offspring generated from current offspring + parents
            # Then selection with roulette or tournament, or score-proportionate
            if self.custom_env['selection'] in ['pop', "pareto"]:
                pq = population + offspring
            elif self.custom_env['selection'] == 'pop&arch':
                pq = population + collection
            else:
                pq = collection

            # # parents will record the changes for every parents
            # # {parent : change in curiosity}
            # parents = {}
            for ind in pq:
                # DONE only add from pq (defined later)
                if self.custom_env['container'] == 'grid':
                    grid.add_to_grid(ind, ind.fit, curio=curio, toolbox=toolbox,
                                            dim=self.custom_env['dim_grid'], 
                                            min_v=self.custom_env['grid_min_v'], 
                                            max_v=self.custom_env['grid_max_v'])
            # update archive and novelty here, needs to be done before selection
            if self.custom_env['container'] == 'archive':
                archive = Archive.update_score(pq, offspring, archive, curio=curio, toolbox=toolbox,
                                k=self.custom_env['nov_k'],
                                add_strategy=self.custom_env['nov_add_strategy'],
                                _lambda=self.custom_env['nov_lambda'])
            
            ## Update the parents' curiosity
            for ind in container[self.custom_env["container"]].pop:
                if hash_ind(ind) in parents.keys() and parents[hash_ind(ind)] in curio.keys():
                    # ind.curiosity = max(0, ind.curiosity + curio[parents[hash_ind(ind)]])
                    ind.curiosity += curio[parents[hash_ind(ind)]]

            print("FITNESS")
            print(*[ind.fit for ind in collection])
            print("NOV")
            print(*[ind.novelty for ind in collection])
            print("CURIOSITY")
            print(*[ind.curiosity for ind in collection])
            print("PARENTS")
            print(parents)
            # print(*[curio[v] for v in parents.values()])
            coll_in_parent = sum([1 for ind in collection if hash_ind(ind) in parents.values()])
            print("IND in PARENTS:", coll_in_parent, "/", len(collection), "in collection")
            # parents_upd_in_coll = [ind.curiosity for ind in parents.values() if ind in collection and ind.curiosity != 0]
            # print("Parents in collection with curiosity != 0:")
            # print(parents_upd_in_coll)

            # Log metrics
            sum_quality = sum([ind.fit for ind in collection])
            sum_novelty = sum([ind.novelty for ind in collection])
            max_quality = min([ind.fit for ind in collection]) # minimize the fitness
            # max_quality=-math.inf
            progress.write(str(len(collection))+" "+ str(sum_quality)+" "+ str(sum_novelty) + " " + str(max_quality)+"\n")
            progress.flush() 

            ## Update fitness.values for nsga2
            for ind in pq:
                if (self.custom_env['quality']=="FIT+NS"):
                    ind.fitness.values=(ind.fit,ind.novelty)
                elif (self.custom_env['quality']=="FIT"):
                    ind.fitness.values=(ind.fit,)
                elif (self.custom_env['quality']=="NS"):
                    ind.fitness.values=(ind.novelty,)
                elif (self.custom_env['quality']=="NSLC"):
                    ind.fitness.values=(ind.novelty,ind.lc)
                elif (self.custom_env['quality']=="curiosity"):
                    ind.fitness.values=(ind.curiosity,)
                # sum_quality= sum_quality+ ind.fit
                # sum_novelty= sum_novelty+ ind.novelty
                if max_quality < fit:
                  max_quality = fit
                #print("Fitness values: "+str(ind.fitness.values)+" Fit=%f Nov=%f"%(ind.fit, ind.novelty))


            # DONE adapt here
            ## Case collection + no_sel
            # => generate pop at random
            if self.custom_env['selection'] == 'noselection':
                select_pop = toolbox.population(n=mu)

            ## Case collection + random sel
            # random selection among the collection aka container
            elif self.custom_env['selection'] == 'random':
                select_pop = random.choices(pq, k = min(mu, len(pq)))
            ## Case collection + score
            # Selection with roulette or tournament, or score-proportionate of collection
            ## Case collection + Pareto
            # filter with NSGA-II
            elif self.custom_env['selection'] in ['score', 'pop', 'pareto']:
                if self.custom_env['quality'] == 'FIT':
                    select_pop = random.choices(pq, weights = [x.fit +1 for x in pq], k=min(mu, len(pq)))

                elif self.custom_env['quality'] == 'NS':
                    select_pop = random.choices(pq, weights = [x.novelty +1 for x in pq], k=min(mu, len(pq)))

                elif self.custom_env['quality'] in ["FIT+NS", "NSLC"]:
                    select_pop = toolbox.select(pq, min(mu, len(pq))) 

                elif self.custom_env['quality'] == "curiosity":
                    select_pop = random.choices(pq, weights = [x.curiosity +1 for x in pq], k=min(mu, len(pq)))

            elif self.custom_env['selection'] == 'pop':
                select_pop = population + select_pop

            population = select_pop
            ### old ways ########
            # population[:] = toolbox.select(pq, mu) 
            #####################


            # Update the hall of fame with the generated individuals
            if paretofront is not None:
                paretofront.update(population)
            indexmax, newvaluemax = max(enumerate([i.log[self.custom_env['watch_max']] for i in pq]), key=operator.itemgetter(1))
            if (newvaluemax>=valuemax):

                valuemax=newvaluemax
                #print("Gen "+str(gen)+", new max ! max fit="+str(valuemax)+" index="+str(indexmax)+" BD="+str(pq[indexmax].bd))
                nnfit, nnbd, log = self.eval_nn(pq[indexmax], env, resdir,render=True,dump=True,name="gen%04d"%(gen))
            elif verbose:
                print(str(gen) +" " + str(newvaluemax)+ "  " + str(valuemax))
        fbd.close()
        finfo.close()
        ffit.close()
        progress.close()
        if self.custom_env['container'] == "grid" and verbose:
            grid.stat_grid(resdir, nb_eval, dim=self.custom_env['dim_grid'])
            grid.dump_grid(resdir, dim=self.custom_env['dim_grid'])

        return population, None, paretofront, grid.grid, archive


    def run(self):



        env = gym.make(self.custom_env['gym_name'], **self.custom_env['env_params'])
        env_name = self.get_env_name()
        resdir = "res_" + env_name + "_" + datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        os.mkdir(resdir)

        ngen = self.custom_env['nb_gen']
        lambda_ = self.custom_env['lambda']
        mu = self.custom_env['mu']

        with open(resdir+"/run_params.log", "w") as rf:
            rf.write("env_name: " + env_name)
            for k in self.custom_env.keys():
                rf.write(k+": "+str(self.custom_env[k])+"\n")
        pop, logbook, paretofront, grid, archive = self.launch_ea(env, mu=mu, lambda_=lambda_, ngen=ngen, resdir=resdir)
        cdir="completed_runs"
        
        try:
            os.mkdir(cdir)
        except FileExistsError:
            pass
        os.rename(resdir,cdir+"/"+resdir) 

        env.close()


        print("Results saved in "+cdir+"/"+resdir)
        no_collision = 0
        for ind in grid.values():
            if ind.log["collision"]== 1000:
                no_collision += 1
        # print("nb cellule without collisions == ", no_collision)

        return pop, logbook, paretofront, grid, archive

def main():
    args = sys.argv[1:]
    if args[0] in ['help', 'h']:
        msg = "To specify parameters, write keyword=value for each parameter\n\
        Example:\n\
            python experiment.py container=grid nb_gen=10\n\nAvailable parameters:\n"
        msg += "container=grid or archive\n"
        msg += "selection=noselection or random, pareto, score, pop, pop&arch\n"
        msg += "quality=NS or FIT+NS, FIT, NSLC, curiosity\n"
        msg += "\n"
        msg += \
"'nb_gen': 200, number of generations\n\
'mu': 100, population size\n\
'lambda': 200, number of individuals generated\n\
'nov_k': 15, k parameter of novelty search\n\
'nov_add_strategy': 'random', archive addition strategy (either 'random' or 'novel' or 'Cully')\n\
'nov_lambda': 6, number of individuals added to the archive\n\nnowarning : hides warnings"
        print(msg)
        return
    ### In Cully2017,
    # nb ite = 50 000
    # batch size = 200

    ## We don't necessarily care about the details of mutation
    # They used a mutation rate between 5% and 12.5%
    # No crossover
    # polynomial mutation for experiment 1, then random new value for exp 2 and 3
    
    ## Grid
    # size = 100 * 100
    # sub-grid depth = +- 3 cells

    ## Archive
    # l = 0.01
    # eps = 0.1
    # k = 15

    ## NSLC variant
    # rho_init = 0.01
    # k = 15

    # Being able to change container archive or grid
    # Being able to change selection op (noselection, random, pareto, 
    # score_based, pop_based, pop & archive based)

    # Being able to change considered value between fitness, novelty, curiosity, 
    # novelty & local quality
    exp = Experiment(cont="archive", sel="random", val="NS")
    exp.set_add_strategy("random")

    # ngen = self.custom_env['nb_gen']
    # lambda_ = self.custom_env['lambda']
    # mu = self.custom_env['mu']
    # exp.set_parameters({'nb_gen' : 1000, 'mu' : 200, 'lambda' : 200})

        # 'min_value': -30, # min genotype value
        # 'max_value': 30, # max genotype value
        # 'min_strategy': 0.5, # min value for the mutation
        # 'max_strategy': 3, # max value for the mutation
        # 'nb_gen': 200, # number of generations
        # 'mu': 100, # population size
        # 'lambda': 200, # number of individuals generated # TODO isn't it supposed to be equal to mu?
        # 'nov_k': 15, # k parameter of novelty search
        # 'nov_add_strategy': "random", # archive addition strategy (either 'random' or 'novel' or 'Cully')
        # 'nov_lambda': 6, # number of individuals added to the archive

    for arg in args:
        keywords = arg.split("=")
        key = keywords[0]
        if key == "nowarning":
            warnings.filterwarnings(action='ignore')
            continue
        word = keywords[1]
        if key in ["container", "selection", "quality", 'nov_add_strategy']:
            exp.set_parameters({key : word})
        elif key in ['nb_gen', 'mu', 'lambda', 'nov_k', 'nov_lambda']:
            exp.set_parameters({key : int(word)})
        else:
            warnings.warn("Unkown or unimplemented key: " + key)
        print("set", key, "to", word)


    pop, logbook, paretofront, grid, archive = exp.run()
    if archive != None:
        print(archive.size())

if __name__ == "__main__":
    main()