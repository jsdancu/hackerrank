class Difference:
    def __init__(self, a):
        self.__elements = a

    # Add your code here
    def computeDifference(self):
        arr_diff = []  
        for i in range(len(self.__elements)-1):
            for j in range(i+1, len(self.__elements)):
                arr_diff.append(abs(self.__elements[i]-self.__elements[j]))
                
        self.maximumDifference = max(arr_diff)

# End of Difference class

_ = input()
a = [int(e) for e in input().split(' ')]

d = Difference(a)
d.computeDifference()

print(d.maximumDifference)