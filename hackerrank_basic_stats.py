import statistics
import math

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    
    mean = round(statistics.mean(arr), 1)
    print(mean)
    median = round(statistics.median(arr), 1)
    print(median)
    mode = statistics.mode(sorted(arr))
    print(mode)
    stdev = round(statistics.pstdev(arr), 1)
    print(stdev)
    t = 1.96
    cl_low = mean - t * stdev/math.sqrt(n)
    cl_high = mean + t * stdev/math.sqrt(n)
    print(f'{cl_low:.1f} {cl_high:.1f}')