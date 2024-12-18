#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))
    
    # s = ''
    # for i in range(n-1, -1, -1):
    #     s += str(arr[i]) + ' '    
    # print(s)

    print(' '.join(str(arr[i]) for i in range(n-1, -1, -1)))