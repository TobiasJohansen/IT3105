import csv
import numpy as np
import tflowtools as tft

class Casemanager():
    def __init__(self, path, readfunction):
        readfunction(self, path)
    
    def read_csv(self, path):
        with open(path) as file:
            cases = np.array([row for row in csv.reader(file, delimiter=";")])
            self.inputs = cases[:,:-1]
            self.targets = [int(target) - 3 for target in cases[:,-1]]
            self.one_hot_targets = [tft.int_to_one_hot(target, 6) for target in self.targets]