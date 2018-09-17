import csv
import numpy as np
import tflowtools as tft

Parity = lambda **param : tft.gen_all_parity_cases(**param)

def Symmetry(**param):
    cases = tft.gen_symvect_dataset(**param)
    cases = set_targets(cases, -1)
    unique_cases = get_unique_targets(cases)
    cases = targets_to_one_hot_vectors(cases, unique_cases)
    return cases

def Wine():
    cases = read_csv("data/winequality_red.txt", ";")
    cases = set_targets(cases, -1)
    unique_cases = get_unique_targets(cases)
    cases = targets_to_one_hot_vectors(cases, unique_cases)
    return cases

def set_targets(cases, target_index):
    for i, case in enumerate(cases):
        target = int(case.pop(target_index))
        cases[i] = [case, target]
    return cases

def get_unique_targets(cases):
    return list(set(np.array(cases)[:,1]))

def targets_to_one_hot_vectors(cases, unique_targets):
    nr_of_targets = len(unique_targets)
    for i, case in enumerate(cases):
        cases[i][1] = tft.int_to_one_hot(unique_targets.index(case[1]), nr_of_targets)
    return cases

def read_csv(path, delimiter):
    with open(path) as file:
        return [row for row in csv.reader(file, delimiter=delimiter)]