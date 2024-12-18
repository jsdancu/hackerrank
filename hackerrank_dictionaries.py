if __name__ == '__main__':
    n = int(input())
    phonebook = {}
    for i in range(n):
        pair = input().rstrip().split()
        phonebook[pair[0]] = pair[1]
        
    while True:
        try:
            name = input()
            # Perform your operations
            if name in phonebook.keys():
                print(f'{name}={phonebook[name]}')
            else:
                print('Not found')
        except EOFError:
            # You can denote the end of input here using a print statement
            break