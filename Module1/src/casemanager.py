import numpy as np
import tools.tflowtools as tft

class Casemanager():
    def __init__(self, datasource, case_fraction, validation_fraction, test_fraction):
        
        cases = datasource["function"](**datasource["parameters"])
        
        self.number_of_features = len(cases[0][0])
        self.number_of_classes = len(cases[0][1])

        np.random.shuffle(cases)
        cases = cases[:round(len(cases) * case_fraction)]

        train_fraction = 1 - (validation_fraction + test_fraction)
        separator1 = round(len(cases) * train_fraction)
        separator2 = separator1 + round(len(cases) * validation_fraction)
        
        self.train_cases = cases[:separator1]
        self.validation_cases = cases[separator1:separator2]
        self.test_cases = cases[separator2:]

    def get_minibatch(self, cases, iteration, minibatch_size):
        nr_of_cases = len(cases)
        start = iteration * minibatch_size % nr_of_cases
        end = start + minibatch_size
        return cases[start:end] + cases[:max(end - nr_of_cases, 0)]

    def get_one_hot_vectors_as_ints(self, one_hot_vectors):
        return [tft.one_hot_to_int(one_hot_vector) for one_hot_vector in one_hot_vectors]