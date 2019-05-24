__author__ = 'Cila'
import numpy as np
import pymssql
import math
import matplotlib.pyplot as plt

a = 0
b = 0
min = 10000000
for i in range(80,120):
    for j in range(80,120):
        temp = 2116668 - (10200 * i + 10002 * j)
        if(temp < min) and temp > 0:
            min = temp
            a = i
            b = j
print("min: %f,511880:%f,511990:%f" %(min,a,b))