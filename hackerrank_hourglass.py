import math
import os
import random
import re
import sys

def hourglass(arr):
    #max_hourglass = 0
    sum_hourglass = []
    for i in range(4):
        for j in range(4):
            #sum_hourglass = arr[i][j]+arr[i][j+1]+arr[i][j+2]+arr[i+1][j+1]+arr[i+2][j]+arr[i+2][j+1]+arr[i+2][j+2]
            sum_hourglass.append(sum(arr[i][j:j + 3]) + arr[i+1][j+1] + sum(arr[i+2][j:j + 3]))
            # if sum_hourglass>max_hourglass:
            #     max_hourglass = sum_hourglass
    
    return max(sum_hourglass)

if __name__ == '__main__':

    arr = []

    for _ in range(6):
        arr.append(list(map(int, input().rstrip().split())))

    print(hourglass(arr))