import csv
import numpy as np

def get_cases(path):
    with open(path) as file:
        return np.array([row for row in csv.reader(file, delimiter=";")])
    