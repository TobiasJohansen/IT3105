import csv
import numpy as np
import tflowtools as tft

Parity = lambda length, double = True: load_parity(length, double)
Symmetry = lambda length, count : load_symmetry(length, count)
Wine = lambda : load_wine()

def load_parity(length, double):
    return tft.gen_all_parity_cases(length, double)

def load_symmetry(length, count):
    return tft.gen_symvect_cases(length, count)

def load_wine():
    return read_csv("data/winequality_red.txt", ";")

def read_csv(path, delimiter):
    with open(path) as file:
        return np.array([row for row in csv.reader(file, delimiter=delimiter)])
            