# Logistic regression algorithm
from math import e

all_M = []
all_B = []

train_M = []
train_B = []

test_M = []
test_B = []


def data():
    global train_M
    global train_B
    global test_M
    global test_B

    with open("./Data/Cancer_Data.csv", "r") as file:
        for line in file:
            row = line.split(",")[1:]
            if row[0] == "M":
                all_M.append(row)
            elif row[0] == "B":
                all_B.append(row)
            else:
                continue

    for feature_index in range(1, 31):
        values = [float(row[feature_index]) for row in all_M + all_B]
        min_value = min(values)
        max_value = max(values)
        for row in all_M + all_B:
            if max_value != min_value:
                row[feature_index] = ((float(row[feature_index]) - min_value) / (max_value - min_value))
            else:
                row[feature_index] = 0

    split_M = int(len(all_M) * 0.8)
    split_B = int(len(all_B) * 0.8)
    train_M = all_M[:split_M]
    train_B = all_B[:split_B]
    test_M = all_M[split_M:]
    test_B = all_B[split_B:]


def sigmoid(x):
    if x < -500:
        return 0.0
    if x > 500:
        return 1.0
    return 1.0 / (1.0 + e**(-x))


def likelihood(k, b):
    probability_product = 1
    for row in train_M:
        score = 0
        for feature_index in range(1, len(row)):
            score += k[feature_index - 1] * float(row[feature_index])
        probability = sigmoid(-score - b)
        probability_product *= probability**0 * (1 - probability)**1
    for row in train_B:
        score = 0
        for feature_index in range(1, len(row)):
            score += k[feature_index - 1] * float(row[feature_index])
        probability = sigmoid(-score - b)
        probability_product *= probability**1 * (1 - probability)**0
    return probability_product


def gradient_k(alpha):
    global k
    for weight_index in range(len(k)):
        gradient = 0
        for row in train_M:
            score = 0
            for feature_index in range(len(k)):
                score += k[feature_index] * float(row[feature_index + 1])
            gradient += 0 * float(row[weight_index + 1])
            gradient -= sigmoid(score + b) * float(row[weight_index + 1])
        for row in train_B:
            score = 0
            for feature_index in range(len(k)):
                score += k[feature_index] * float(row[feature_index + 1])
            gradient += 1 * float(row[weight_index + 1])
            gradient -= sigmoid(score + b) * float(row[weight_index + 1])
        k[weight_index] += gradient * alpha


def gradient_b(alpha):
    global b
    gradient = 0
    for row in train_M:
        score = 0
        for feature_index in range(len(k)):
            score += k[feature_index] * float(row[feature_index + 1])
        gradient += 0
        gradient -= sigmoid(score + b)
    for row in train_B:
        score = 0
        for feature_index in range(len(k)):
            score += k[feature_index] * float(row[feature_index + 1])
        gradient += 1
        gradient -= sigmoid(score + b)
    b += gradient * alpha


k = [0] * 30
b = 1


def logistic_regression(alpha, count):
    for i in range(count):
        gradient_k(alpha)
        gradient_b(alpha)

    correct_count = 0
    for row in test_M:
        score = 0
        for feature_index in range(1, len(row)):
            score += k[feature_index - 1] * float(row[feature_index])
        if round(sigmoid(score + b)) == 0:
            correct_count += 1
    for row in test_B:
        score = 0
        for feature_index in range(1, len(row)):
            score += k[feature_index - 1] * float(row[feature_index])
        if round(sigmoid(score + b)) == 1:
            correct_count += 1
    return 100 * correct_count / (len(test_M) + len(test_B))


data()
print(logistic_regression(0.0033, 50))
