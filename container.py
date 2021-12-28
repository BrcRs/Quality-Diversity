from abc import abstractmethod, abstractstaticmethod
from scipy.spatial import KDTree
import random
import numpy as np
from operator import itemgetter

import functools
import math


class Container:
    def __init__(self, lbd, fit_lbd, k=15):
        self.all_bd=lbd
        # self.kdtree=KDTree(self.all_bd)
        self.k=k
        self.fit = fit_lbd

    def update(self,new_bd, fit_bd):
        oldsize=len(self.all_bd)
        self.all_bd=self.all_bd + new_bd
        self.fit += fit_bd
        # self.kdtree=KDTree(self.all_bd)
        #print("Archive updated, old size = %d, new size = %d"%(oldsize,len(self.all_bd)))    

    def size(self):
        return len(self.all_bd)

    @abstractstaticmethod
    def update_score():
        raise NotImplementedError()

class Grid:
    def __init__(self):
        self.grid = dict()
    
    def get_grid_coord(bd, dim=[100, 100], min_v=[0,0], max_v=[1, 1]):
        x,y = bd
        x = dim[0]*x//(max_v[0] - min_v[0])
        y = dim[1]*y//(max_v[1] - min_v[1])
        return (x,y)

    def add_to_grid(self,ind, local_quality, dim=[100, 100], min_v=[0,0], max_v=[1, 1]):
        # determining the grid cell coordinates to add to
        x, y = self.get_grid_coord(ind.bd, dim, min_v, max_v)	
        if (x,y) in self.keys():
            if local_quality < self[(x,y)].fit: # this assumes that the lower the number of collisions, the better
                self[(x,y)] = ind
        else:
            self[(x,y)] = ind

    def get_pop(self):
        return list(self.grid.values())

    def stat_grid(self, resdir, nb_eval, dim=[100, 100]):
        if(len(self.values())==0):
            print("Empty grid: no stats...")
            return

        nb_filled=0
        max_v=None
        for i in range(dim[0]):
            for j in range(dim[1]):
                if ((i,j) in self.keys()):
                    nb_filled+=1
                    
        nbcells=functools.reduce(lambda x, y: x*y, dim)
        c_values=[ind.fit for ind in list(self.values())]
        max_v=max(c_values)
        total_quality=sum(c_values)
        #print("Number of evaluations: %d"%(nb_eval))
        print("Coverage: %.2f %% (%d cells out of %d)"%(float(nb_filled)/float(nbcells)*100., nb_filled, nbcells)+" Max score: %.2f"%(max(c_values))+" Min score: %.2f"%(min(c_values))+" Total quality: %.2f"%(total_quality))
        stat_grid={
            'nb_eval': nb_eval,
            'nb_cells': nbcells,
            'nb_filled': nb_filled,
            'max_score': max(c_values),
            'min_score': min(c_values),
            'coverage': float(nb_filled)/float(nbcells),
            }
        with open(resdir+"/stat_grid.log","a") as sf:
            sf.write(str(stat_grid)+"\n")
            


    def dump_grid(self, resdir, dim=[100, 100]):
        if(len(self.values())==0):
            print("Empty grid: no dump...")
            return
        with open(resdir+"/map.dat","w") as mf:
            for i in range(dim[0]):
                for j in range(dim[1]):
                    if ((i,j) in self.keys()):
                        mf.write("%.2f "%(self[(i,j)].fit))
                    else:
                        mf.write("=== ")
                mf.write("\n")
        with open(resdir+"/map_collisions.dat","w") as mf:
            for i in range(dim[0]):
                for j in range(dim[1]):
                    if ((i,j) in self.keys()):
                        mf.write("%.2f "%(self[(i,j)].log["collision"]))
                    else:
                        mf.write("=== ")
                mf.write("\n")
        with open(resdir+"/map_bd.dat","w") as mf:
            for p in self.keys():
                ind=self[p]
                for i in range(len(ind.bd)):
                    mf.write("%f "%(ind.bd[i]))
                mf.write("\n")


class Archive(Container):
    """Archive used to compute novelty scores."""
    def __init__(self, lbd, fit_lbd, k=15):
        super.__init__(lbd, fit_lbd, k)
        self.kdtree=KDTree(self.all_bd)
        self.pop = []
        #print("Archive constructor. size = %d"%(len(self.all_bd)))
        
    def update(self,new_bd, fit_bd):
        super.update(new_bd, fit_bd)
        self.kdtree=KDTree(self.all_bd)
        #print("Archive updated, old size = %d, new size = %d"%(oldsize,len(self.all_bd)))
    
    def get_pop(self):
        return self.pop

    def get_nov(self,bd,fit_bd, population=[]):
        dpop=[]
        fitpop = []
        for ind in population:
            dpop.append(np.linalg.norm(np.array(bd)-np.array(ind.bd)))
            fitpop.append(ind.fit)
        kNeighboursInf = 0
        darch,ind=self.kdtree.query(np.array(bd),self.k)
        d=dpop+list(darch)
        f = fitpop + [self.fit[i] for i in ind]
        arg_sort = [x for x,y in sorted(enumerate(d), key = lambda x: x[1])]
        d.sort()
        for i in arg_sort[:self.k]:
            if f[i] > fit_bd:
                kNeighboursInf += 1
        if (d[0]!=0):
            print("WARNING in novelty search: the smallest distance should be 0 (distance to itself). If you see it, you probably try to get the novelty with respect to a population your indiv is not in. The novelty value is then the sum of the distance to the k+1 nearest divided by k.")
        return sum(d[:self.k+1])/self.k, kNeighboursInf # as the indiv is in the population, the first value is necessarily a 0.

    
    def update_score(population, offspring, archive, k=15, add_strategy="random", _lambda=6, verbose=False):
        """Update the novelty criterion (including archive update) 

        Implementation of novelty search following (Gomes, J., Mariano, P., & Christensen, A. L. (2015, July). Devising effective novelty search algorithms: A comprehensive empirical study. In Proceedings of GECCO 2015 (pp. 943-950). ACM.).
        :param population: is the set of indiv for which novelty needs to be computed
        :param offspring: is the set of new individuals that need to be taken into account to update the archive (may be the same as population, but it may also be different as population may contain the set of parents)
        :param k: is the number of nearest neighbors taken into account
        :param add_strategy: is either "random" (a random set of indiv is added to the archive) or "novel" (only the most novel individuals are added to the archive).
        :param _lambda: is the number of individuals added to the archive for each generation
        The default values correspond to the one giving the better results in the above mentionned paper.

        The function returns the new archive
        """
        # TODO change population and offspring for something else (a single variable?)
        # Novelty scores updates
        if (archive) and (archive.size()>=k):
            if (verbose):
                print("Update Novelty. Archive size=%d"%(archive.size())) 
            # TODO add curiosity
            for ind in population:
                ind.novelty , ind.lc=archive.get_nov(ind.bd,ind.fit, population)
                #print("Novelty: "+str(ind.novelty))
        else:
            if (verbose):
                print("Update Novelty. Initial step...") 
            # TODO add curiosity
            for ind in population:
                ind.novelty = 0.
                ind.lc = 0
                ind.curiosity = 0.
            
        if (verbose):
            print("Fitness (novelty): ",end="") 
            for ind in population:
                print("%.2f, "%(ind.novelty),end="")
            print("")
        if (len(offspring)<_lambda):
            print("ERROR: updateNovelty, lambda(%d)<offspring size (%d)"%(_lambda, len(offspring)))
            return None

        lbd=[]
        # Update of the archive
        # TODO add all the other strategies
        if(add_strategy=="random"):
            # random individuals are added
            l=list(range(len(offspring)))
            random.shuffle(l)
            if (verbose):
                print("Random archive update. Adding offspring: "+str(l[:_lambda])) 
            lbd=[offspring[l[i]].bd for i in range(_lambda)]
            lbd_fit = [-1*offspring[l[i]].fit for i in range(_lambda)]
            archive.pop += [offspring[l[i]] for i in range(_lambda)]
        elif(add_strategy=="novel"):
            # the most novel individuals are added
            soff=sorted(offspring,lambda x:x.novelty)
            ilast=len(offspring)-_lambda
            lbd=[soff[i].bd for i in range(ilast,len(soff))]
            lbd_fit=[-1*soff[i].fit for i in range(ilast,len(soff))]
            archive.pop += [soff[i] for i in range(ilast,len(soff))]
            if (verbose):
                print("Novel archive update. Adding offspring: ")
                for offs in soff[ilast:len(soff)]:
                    print("    nov="+str(offs.novelty)+" fit="+str(offs.fitness.values)+" bd="+str(offs.bd))
        elif add_strategy == "Cully":
            raise NotImplementedError()
        else:
            the_valid = ["random", "novel", "Cully"]
            print("ERROR: update_score: unknown add strategy(%s), valid alternatives are", the_valid, ""%(add_strategy))
            return None
            
        if(archive==None):
            archive = Archive(lbd,lbd_fit,k)
        else:
            archive.update(lbd,lbd_fit)

        return archive
