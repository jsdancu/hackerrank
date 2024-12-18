#!/bin/python3

import math
import os
import random
import re
import sys

def athlete_sort(k, arr):
    return sorted(arr, key=lambda athlete: athlete[k])

if __name__ == '__main__':
    first_multiple_input = input().rstrip().split()

    n = int(first_multiple_input[0])

    m = int(first_multiple_input[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input().strip())
    if (1<=n) and (n<=1000) and (1<=m) and (m<=1000) and (0<=k) and (k<m) and (max(max(arr))<=1000):
        arr_sorted = athlete_sort(k, arr)
        for i in arr_sorted:
            print('\t'.join(map(str, i)))
    else:
        print("dimensions or elements out of range")