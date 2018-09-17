import csv
import numpy as np
import tflowtools as tft

Parity = lambda **param : tft.gen_all_parity_cases(**param)
Symmetry = lambda **param : tft.gen_symvect_cases(**param)
Wine = lambda **param : read_csv("data/winequality_red.txt", ";")

def read_csv(path, delimiter):
    with open(path) as file:
        return np.array([row for row in csv.reader(file, delimiter=delimiter)])