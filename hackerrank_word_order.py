from collections import OrderedDict
from string import ascii_lowercase

if __name__ == "__main__":
    n = int(input())
    if (1<=n) and (n<=1e5):
        sum = 0
        word_dict = OrderedDict()
        for i in range(n):
            word = str(input())
            if word in word_dict.keys():
                word_dict[word] += 1
            else:
                word_dict[word] = 1

        print(len(word_dict.keys()))
        print('\t'.join(map(str, (word_dict[word] for word in word_dict.keys()))))
    else:
        print("n out of range")