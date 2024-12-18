import math
import os
import random
import re
import sys

def bubble_sort(a):
    total_swaps = 0
    n = len(a)
    for i in range(n):
        swaps = 0
    
        for j in range(n-1):
            if (a[j] > a[j + 1]):
                a[j], a[j+1] = a[j+1], a[j]
                swaps += 1
    
        total_swaps += swaps
        
        if (swaps == 0):
            break
            
    return a, total_swaps

if __name__ == '__main__':
    n = int(input().strip())

    a = list(map(int, input().rstrip().split()))

    # Write your code here
    x, swaps = bubble_sort(a)
    
    print('Array is sorted in', swaps, 'swaps.')
    print('First Element:', x[0])
    print('Last Element:', x[-1])