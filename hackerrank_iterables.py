from itertools import combinations

if __name__ == "__main__":
    n = int(input())
    if (1<=n) and (n<=10):
        string = str(input().replace(' ', ''))
        k = int(input())
        if (1<=k) and (k<=n) and (len(string) == n):
            combinations = list(combinations(string, k))
            counter = 0
            for combo in combinations:
                if "a" in combo:
                    counter += 1
            print(counter/len(combinations))
        else:
            print("string or k out of dimension")
    else:
        print("n out of range")