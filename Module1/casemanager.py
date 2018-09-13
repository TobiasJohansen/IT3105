import numpy as np

class Casemanager():
    def __init__(self, datasource, validation_fraction, test_fraction):
        cases = datasource() if not isinstance(datasource, list) else datasource[0](*datasource[1:])  
        
        print(cases[0])

        exit()
        
        np.random.shuffle(cases)


        train_fraction = 1 - (validation_fraction + test_fraction)
        separator1 = round(len(cases) * train_fraction)
        separator2 = separator1 + round(len(cases) * validation_fraction)
        
        self.train_cases = cases[:separator1]
        self.validation_cases = cases[separator1:separator2]
        self.test_cases = cases[separator2:]