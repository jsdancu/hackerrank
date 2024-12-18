from collections import deque

def piling_up(n, d):
    
    flag = True
    pile = deque()
    while (flag == True) and (len(d)>0):
        if (len(pile)>0):
            if (d[0]>=d[-1]):
                if (d[0]<=pile[-1]):
                    pile.append(d[0])
                    d.popleft()
                else:
                    flag = False
            else:
                if (d[-1]<=pile[-1]):
                    pile.append(d[-1])
                    d.pop()
                else:
                    flag = False
        else:
            if (d[0]>=d[-1]):
                pile.append(d[0])
                d.popleft()
            else:
                pile.append(d[-1])
                d.pop()
        
    if flag == True:
        print('Yes') 
    else:
        print('No')   

if __name__ == '__main__':
    t = int(input())
    if (0 < t) and (t <= 5):
        for i in range(t):
            n = int(input())
            d = deque(map(int, input().rstrip().split()))
            piling_up(n, d)    
    else:
        print('Number of instructions out or range')