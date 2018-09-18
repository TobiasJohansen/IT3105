import csv
import numpy as np
import tools.tflowtools as tft

parity = lambda **param : tft.gen_all_parity_cases(**param)
symmetry = lambda **param : set_one_hot_vectors(tft.gen_symvect_dataset(**param), -1)
wine = lambda : set_one_hot_vectors(read_csv("data/winequality_red.txt", ";"), -1)

def set_one_hot_vectors(cases, target_index):
    unique_targets = list(set(np.array(cases)[:,1]))
    nr_of_unique_targets = len(unique_targets)
    for i, case in enumerate(cases):
        one_hot_vector = tft.int_to_one_hot(unique_targets.index(int(case.pop(target_index))), nr_of_unique_targets)
        cases[i] = [case, one_hot_vector]

def read_csv(path, delimiter):
    with open(path) as file:
        return [row for row in csv.reader(file, delimiter=delimiter)]