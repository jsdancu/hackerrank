import math
import os
import random
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def battery_life(laptopCharged, laptopLasted, timeCharged):
    indices_kept = (laptopLasted < 8.0)
    laptopCharged = laptopCharged[indices_kept]
    laptopLasted = laptopLasted[indices_kept]

    reg = LinearRegression().fit(laptopCharged.reshape(-1, 1), laptopLasted.reshape(-1, 1))
    predicted = reg.predict([[timeCharged]])
    return round(predicted[0][0], 2)


if __name__ == '__main__':
    timeCharged = float(input().strip())

    if timeCharged > 4.0:
        print(8.0)

    else:
        laptopCharged = np.array([])
        laptopLasted = np.array([])

        with open(os.path.join("trainingdata.txt")) as f:
            for line in f:
                l = list(map(float, line.rstrip().split(','))) 
                laptopCharged = np.append(laptopCharged, l[0])
                laptopLasted = np.append(laptopLasted, l[1])

        '''
        plt.scatter(laptopCharged, laptopLasted)
        plt.xlabel("time charged")
        plt.ylabel("time lasted")
        plt.title("Laptop battery time charged vs lasted")
        #plt.show()
        plt.savefig("battery_life.pdf")
        '''

        print(battery_life(laptopCharged, laptopLasted, timeCharged))

    
