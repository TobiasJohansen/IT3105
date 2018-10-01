import csv
import data.mnist.mnist_basics as mb
import numpy as np
import tools.tflowtools as tft

parity = lambda **param : to_np(restructure(set_int_targets(tft.gen_all_parity_cases(**param))))
symmetry = lambda **param : to_np(restructure(set_both_targets(tft.gen_symvect_dataset(**param))))
autoencoder = lambda **param : to_np(restructure(set_int_targets(tft.gen_all_one_hot_cases(**param))))
bit_counter = lambda **param : to_np(restructure(set_int_targets(tft.gen_vector_count_cases(**param))))
segment_counter = lambda **param : to_np(restructure(set_int_targets(tft.gen_segmented_vector_cases(**param))))
mnist = lambda : to_np(normalize(restructure(set_both_targets(mb.load_all_flat_cases(dir="data/mnist/", unify=True))), 255))
wine_quality = lambda : to_np(normalize(restructure(set_both_targets(read_csv("data/winequality_red.txt", ";")))))
glass = lambda : to_np(normalize(restructure(set_both_targets(read_csv("data/glass.txt", ",")))))
yeast = lambda : to_np(normalize(restructure(set_both_targets(read_csv("data/yeast.txt", ",")))))
poker_hand = lambda : to_np(normalize(restructure(set_both_targets(read_csv("data/poker_hand.txt", ",")))))

def to_np(cases):
    return np.array(cases)

def normalize(cases, range = None):
    features = np.array(cases[0], dtype="d")
    if range:
        features /= range
    else:
        features -= np.min(features, axis=0)
        features /= np.ptp(features, axis=0)
    return np.array([features.tolist(), cases[1], cases[2]])

def restructure(cases):
    return np.array(cases).T.tolist()

def set_int_targets(cases):
    for i, case in enumerate(cases):
        cases[i].append(tft.one_hot_to_int(case[1]))
    return cases

def set_both_targets(cases):
    classes = list(sorted(set(np.array(cases)[:,-1])))
    nr_of_classes = len(classes)
    for i, case in enumerate(cases):
        int = classes.index(case.pop(-1))
        one_hot_vector = tft.int_to_one_hot(int, nr_of_classes)
        cases[i] = [case, one_hot_vector, int]
    return cases

def read_csv(path, delimiter):
    with open(path) as file:
        return [row for row in csv.reader(file, delimiter=delimiter)]