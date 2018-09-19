import csv
import data.mnist.mnist_basics as mb
import numpy as np
import tools.tflowtools as tft

parity = lambda **param : tft.gen_all_parity_cases(**param)
symmetry = lambda **param : set_one_hot_vectors(set_targets(tft.gen_symvect_dataset(**param), -1))
wine = lambda : set_one_hot_vectors(set_targets(read_csv("data/winequality_red.txt", ";"), -1))
mnist = lambda : mb.load_all_flat_cases(unify=True)

def set_targets(cases, target_index):
    for i, case in enumerate(cases):
        target = case.pop(target_index)
        cases[i] = [case, target]
    return set_one_hot_vectors(cases)

def set_one_hot_vectors(cases):
    classes = list(sorted(set(np.array(cases)[:,1])))
    nr_of_classes = len(classes)
    for i, case in enumerate(cases):
        one_hot_vector = tft.int_to_one_hot(classes.index(case[1]), nr_of_classes)
        cases[i] = [case[0], one_hot_vector]
    return cases

def read_csv(path, delimiter):
    with open(path) as file:
        return [row for row in csv.reader(file, delimiter=delimiter)]

print(mnist())