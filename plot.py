import matplotlib.pyplot as plt
import numpy as np

def recup(fichier, nb_gen):
  size=[]
  max_qual=[]
  sum_nov=[]
  sum_qual=[]
  f = open(fichier, "r")
  myline = f.readline()
  i = 0
  while myline != '' and i < nb_gen:
    myline = f.readline()
    if ("##" not in myline) and (myline != ''):
      x= myline.split(' ')
    #   print (len(x),x)
      size.append(x[0])
      sum_qual.append(x[1])
      sum_nov.append(x[2])
      max_qual.append(x[3])
      i += 1

  return size,sum_qual,sum_nov,max_qual

def main():

    filenames = [
        ("progress-archive-nosel (2).log", "no_selection"), 
        ("progress-archive-pareto-NSLC (2).log", "NSLC"),
        ("progress-archive-pop-curiosity.log", "pop_curiosity"),
        ("progress-archive-pop-FIT.log", "pop_fitness"),
        ("progress-archive-pop-NS.log", "pop_novelty"),
        ("progress-archive-random.log", "random"),
        ("progress-archive-score-curiosity.log", "curiosity"),
        ("progress-archive-score-FIT.log", "fitness"),
        ("progress-archive-score-NS.log", "novelty"),
        ]
    nb_gen = 1000
    data = {"size":{}, "sum_qual":{}, "sum_nov":{}, "max_qual":{}}
    x = range(nb_gen)
    for name in filenames:
        size,sum_qual,sum_nov,max_qual = recup(name[0], nb_gen)
        # try:
        data['size'][name[1]] = [int(s) for s in size]
        data['sum_qual'][name[1]] = [-float(s) for s in sum_qual] #sum_qual.copy()
        data['sum_nov'][name[1]] = [float(s) for s in sum_nov]#sum_nov.copy()
        data['max_qual'][name[1]] = [-float(s) for s in max_qual]#max_qual.copy()
        # except ValueError:
        #     print(size)

        # print(data['size'][name[1]])
    # for metric in ['size', 'sum_qual', 'sum_nov', 'max_qual']:
    #     for name in filenames:
    #         plt.plot(x, data[metric][name[1]], label=name[1])
    #     plt.xlabel('Number of Iterations')
    #     # plt.ylabel('y label')
    #     plt.title(metric)
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()
    metric = 'size'
    for name in filenames:
        plt.plot(x, data[metric][name[1]], label=name[1])
    plt.xlabel('Number of Iterations')
    # plt.set_xticks(range(0, nb_gen, 100))
    # plt.ylabel('y label')
    plt.title(metric)
    plt.legend()
    plt.grid(True)
    plt.show()

    metric = 'sum_qual'
    for name in filenames:
        plt.plot(x, data[metric][name[1]], label=name[1])
    plt.xlabel('Number of Iterations')
    # plt.set_xticks(range(0, nb_gen, 100))
    # plt.ylabel('y label')
    plt.title(metric)
    plt.legend()
    plt.grid(True)
    plt.show()

    metric = 'sum_nov'
    for name in filenames:
        plt.plot(x, data[metric][name[1]], label=name[1])
    plt.xlabel('Number of Iterations')
    # plt.set_xticks(range(0, nb_gen, 100))
    # plt.ylabel('y label')
    plt.title(metric)
    plt.legend()
    plt.grid(True)
    plt.show()

    metric = 'max_qual'
    for name in filenames:
        plt.plot(x, data[metric][name[1]], label=name[1])
    plt.xlabel('Number of Iterations')
    # plt.set_xticks(range(0, nb_gen, 100))
    # plt.ylabel('y label')
    plt.title(metric)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
