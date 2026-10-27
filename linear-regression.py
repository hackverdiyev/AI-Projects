# Linear regression algorithm
from math import inf

all_rows = []
train = []
test = []
max_of_all = [0, 0]


def data():
    global all_rows
    global train
    global test

    with open("./Data/Salary_Data.csv", "r") as file:
        for line in file:
            row = line.split(",")
            if row[0].isalpha():
                continue
            all_rows.append(row)
            for column in range(2):
                if max_of_all[column] < float(row[column]):
                    max_of_all[column] = float(row[column])

    split_index = int(len(all_rows) * 0.7)
    train = all_rows[:split_index]
    test = all_rows[split_index:]


def MSE(dataset, k, b):
    total_error = 0
    for row in dataset:
        total_error += (float(row[1]) / max_of_all[1] - (k * float(row[0]) / max_of_all[0] + b)) ** 2
    return total_error / len(dataset)


def gradient_k(dataset, k, b):
    total_gradient = 0
    for row in dataset:
        x = float(row[0]) / max_of_all[0]
        y = float(row[1]) / max_of_all[1]
        total_gradient += -2 * y * x + 2 * k * x**2 + 2 * x * b
    return total_gradient / len(dataset)


def gradient_b(dataset, k, b):
    total_gradient = 0
    for row in dataset:
        x = float(row[0]) / max_of_all[0]
        y = float(row[1]) / max_of_all[1]
        total_gradient += -2 * y + 2 * k * x + 2 * b
    return total_gradient / len(dataset)


def linear_regression(alpha, count):
    k, b = 2, 2
    best_parameters = (0, 0)
    min_error = inf
    for i in range(count):
        dk = gradient_k(train, k, b)
        db = gradient_b(train, k, b)
        k = k - alpha * dk
        b = b - alpha * db
        error = MSE(train, k, b)
        if error < min_error:
            min_error = error
            best_parameters = (k, b)
    return MSE(test, best_parameters[0], best_parameters[1])


data()
print(linear_regression(0.0033, 100))
