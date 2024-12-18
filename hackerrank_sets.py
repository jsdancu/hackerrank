def average(array):
    # your code goes here
    sum_arr = sum(set(array))
    total_arr = len(set(array))
    average = round(sum_arr/total_arr, 3)
    return average

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)