# Quality Diversity framework

## Introduction
This project aims to reproduce the Quality Diversity framework defined by Antoine Cully and Yiannis Demiris in their paper Cully, A., & Demiris, Y. (2017). Quality and Diversity Optimization: A Unifying Modular Framework. IEEE Transactions on Evolutionary Computation) and use it for the navigation problem in a Stanley's labyrinth.

## Installation

First, clone this repository:

  git clone https://github.com/BrcRs/Quality-Diversity.git
  
Then, install the downloaded dependencies:

    cd Quality-Diversity/dependencies ; tar -xf fastsim_gym.tar.gz
    cd Quality-Diversity/dependencies ; tar -xf libfastsim.tar.gz
    cd Quality-Diversity/dependencies ; tar -xf pyfastsim.tar.gz
 
    cd Quality-Diversity/dependencies/libfastsim/ ; python2 ./waf configure
    cd Quality-Diversity/dependencies/libfastsim/ ; python2 ./waf build
    cd Quality-Diversity/dependencies/libfastsim/ ; python2 ./waf install
  
    cd Quality-Diversity/dependencies/pyfastsim ; pip3 install .
    cd Quality-Diversity/dependencies/fastsim_gym/ ; pip3 install .
  
Additional dependencies:

    pip3 install plot
    pip3 install cma
    pip3 install deap
    pip install scoop # Note : scoop won't work with this project

## Usage

Show the available parameters with:

    python3 experiment.py help

You can then launch your own experiments, for instance:

    python3 experiment.py container=archive selection=score quality=curiosity nb_gen=1000 nov_add_strategy=Cully archive_density=6 nowarning

The experiment will produce a file named `progress.log` which, for each iteration, shows the size of the collection (whether that be archive or grid), the sum of all fitness, the sum of all novelty and the best fitness found in the current collection.

## Authors
B. Rose (@BrcRs)     
G. Amairi (@ghada-source)
