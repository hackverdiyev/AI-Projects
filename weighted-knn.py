# KNN algorithm with cancer data: 98.2% accuracy (k=7)
from math import sqrt, inf

all_M = []
all_B = []

train_M = []
train_B = []

test_M = []
test_B = []

max_of_all = [0] * 30


def data():
    global train_M
    global train_B
    global test_M
    global test_B
    global max_of_all

    with open("./Data/Cancer_Data.csv", "r") as file:
        for line in file:
            row = line.split(",")[1:]
            if row[0] == "M":
                all_M.append(row)
            elif row[0] == "B":
                all_B.append(row)
            else:
                continue
            for feature_index in range(30):
                if max_of_all[feature_index] < float(row[feature_index + 1]):
                    max_of_all[feature_index] = float(row[feature_index + 1])

    split_M = int(len(all_M) * 0.8)
    split_B = int(len(all_B) * 0.8)
    train_M = all_M[:split_M]
    train_B = all_B[:split_B]
    test_M = all_M[split_M:]
    test_B = all_B[split_B:]


def test(row, k):
    neighbors = [(inf, "X")] * k
    for dataset, label in ((train_M, "M"), (train_B, "B")):
        for training_row in dataset:
            distance = 0
            for feature_index in range(30):
                distance += ((float(row[feature_index + 1]) - float(training_row[feature_index + 1])) / max_of_all[feature_index]) ** 2
            distance = sqrt(distance)
            if distance < neighbors[-1][0]:
                neighbors.pop()
                neighbors.append((distance, label))
                neighbors = sorted(neighbors)

    malignant_weight = 0
    benign_weight = 0
    for neighbor_index in range(k):
        if neighbors[neighbor_index][1] == "M":
            malignant_weight += 1 / neighbors[neighbor_index][0]
        if neighbors[neighbor_index][1] == "B":
            benign_weight += 1 / neighbors[neighbor_index][0]
    if malignant_weight > benign_weight:
        return "M"
    return "B"


def accuracy(k):
    total_count = len(test_M) + len(test_B)
    correct_count = 0

    for row in test_M:
        if test(row, k) == "M":
            correct_count += 1

    for row in test_B:
        if test(row, k) == "B":
            correct_count += 1

    return correct_count * 100 / total_count


data()
print(accuracy(7))
