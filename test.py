#!/usr/bin/env python3

import numpy as np


def convoluntional_calculation(input, kernel):
    size_0 = input.shape[0] - kernel.shape[0] + 1
    size_1 = input.shape[1] - kernel.shape[1] + 1
    result = np.zeros((size_0, size_1))
    for i in range(size_0):
        for j in range(size_1):
            result[i, j] = np.sum(input[i : i + kernel.shape[0], j : j + kernel.shape[1]] * kernel)
    return result


input = np.array([[2, 2, 2, 2], [4, 4, 4, 4], [8, 8, 8, 8]])
kernel = np.array([[1, 1], [2, 2]])
print(convoluntional_calculation(input, kernel))
