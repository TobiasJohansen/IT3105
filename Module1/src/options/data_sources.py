import csv
import data.mnist.mnist_basics as mb
import numpy as np
import tools.tflowtools as tft

parity = lambda **param : split_inputs_and_targets(tft.gen_all_parity_cases(**param))
symmetry = lambda **param : split_inputs_and_targets(set_one_hot_vectors(tft.gen_symvect_dataset(**param), -1))
wine = lambda : split_inputs_and_targets(set_one_hot_vectors(read_csv("data/winequality_red.txt", ";"), -1))
mnist = lambda : split_inputs_and_targets(set_one_hot_vectors(mb.load_all_flat_cases(dir="data/mnist/", unify=True), -1))

def split_inputs_and_targets(cases):
    split_cases = np.ndarray((len(cases), len(cases[0])), dtype=object)
    split_cases[:] = cases
    return split_cases.T

def set_one_hot_vectors(cases, target_index):
    classes = list(sorted(set(np.array(cases)[:,target_index])))
    nr_of_classes = len(classes)
    for i, case in enumerate(cases):
        one_hot_vector = tft.int_to_one_hot(classes.index(case.pop(target_index)), nr_of_classes)
        cases[i] = [case, one_hot_vector]
    return cases

def read_csv(path, delimiter):
    with open(path) as file:
        return [row for row in csv.reader(file, delimiter=delimiter)]