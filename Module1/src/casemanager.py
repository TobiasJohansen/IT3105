import random
import tools.tflowtools as tft

class Casemanager():
    def __init__(self, datasource, case_fraction, validation_fraction, test_fraction):
        
        all_cases = datasource["function"](**datasource["parameters"])
        cases = self.get_randomised_subset(all_cases, round(len(all_cases[0]) * case_fraction))
        
        self.number_of_features = len(cases[0][0])
        self.number_of_classes = len(cases[1][0])
        
        number_of_cases = len(cases[0])
        train_fraction = 1 - (validation_fraction + test_fraction)
        validation_start = round(number_of_cases * train_fraction)
        test_start = validation_start + round(number_of_cases * validation_fraction)

        self.train_cases = cases[:,:validation_start]
        self.validation_cases = cases[:,validation_start:test_start]
        self.test_cases = cases[:,test_start:]

    def get_randomised_subset(self, cases, size):
        number_of_cases = len(cases[0])
        randomized_indices = random.sample(range(0, number_of_cases), min(size, number_of_cases))
        return cases[:,randomized_indices]

    def get_minibatch(self, minibatch_size):
        return self.get_randomised_subset(self.train_cases, minibatch_size).tolist()

    def one_hot_vectors_to_ints(self, one_hot_vectors):
        return [tft.one_hot_to_int(one_hot_vector) for one_hot_vector in one_hot_vectors]

    def get_train_cases(self): return self.train_cases.tolist()
    def get_validation_cases(self): return self.validation_cases.tolist()
    def get_test_cases(self): return self.test_cases.tolist()