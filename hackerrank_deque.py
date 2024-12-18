from collections import deque

def deque_functions(d, command, k=0):
    
    if (command == 'append'):
        d.append(k)
    elif (command == 'appendleft'):
        d.appendleft(k)
    elif (command == 'pop'):
        d.pop()
    elif (command == 'popleft'):
        d.popleft()
    elif (command == 'extend'):
        d.extend(k)
    elif (command == 'remove'):
        d.remove(k)
    elif (command == 'reverse'):
        d.reverse()
    elif (command == 'rotate'):
        d.rotate(k)
        
    return d

if __name__ == '__main__':
    n = int(input())
    if (0 < n) and (n <=100):
        d = deque()
        for i in range(n):
            string = input()
            if len(string.split()) > 1:
                command, k = string.split()
                deque_functions(d, command, k)
            else:
                command = string
                deque_functions(d, command)
        print(*d)
    else:
        print('Number of instructions out or range')