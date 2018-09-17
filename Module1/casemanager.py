import numpy as np

class Casemanager():
    def __init__(self, datasource, case_fraction, validation_fraction, test_fraction):
        
        cases = datasource["function"](**datasource["parameters"])
        
        self.number_of_inputs = len(cases[0][0])
        self.number_of_classes = len(cases[0][1])

        np.random.shuffle(cases)
        cases = np.array(cases[:round(len(cases) * case_fraction)])

        train_fraction = 1 - (validation_fraction + test_fraction)
        separator1 = round(len(cases) * train_fraction)
        separator2 = separator1 + round(len(cases) * validation_fraction)
        
        self.train_cases = cases[:separator1]
        self.validation_cases = cases[separator1:separator2]
        self.test_cases = cases[separator2:]