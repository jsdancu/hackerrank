def happiness(n, m, array, a, b):
    happiness = 0
    for i in array:
        if i in a:
            happiness += 1
        elif i in b:
            happiness -= 1
    
    return happiness

if __name__ == '__main__':
    n, m = map(int, input().rstrip().split())
    array = list(map(int, input().split()))
    a = set(list(map(int, input().split())))
    b = set(list(map(int, input().split())))
    
    print(happiness(n, m, array, a, b))