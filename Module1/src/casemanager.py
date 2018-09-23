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
        
        validation_cases = cases[separator1:separator2]
        validation_inputs, validation_targets = map(list, np.array(validation_cases).T)
        validation_targets_as_ints = [tft.one_hot_to_int(one_hot_vector) for one_hot_vector in validation_targets]
        self.validation_cases = [validation_inputs, validation_targets, validation_targets_as_ints]

        self.test_cases = cases[separator2:]

    def get_minibatch(self, minibatch_size):
        inputs, targets = map(list, np.array(self.train_cases[:minibatch_size]).T)
        np.random.shuffle(self.train_cases)
        return inputs, targets