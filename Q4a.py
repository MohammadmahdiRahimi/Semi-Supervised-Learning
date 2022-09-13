import numpy as np
import matplotlib.pyplot as plt

#function to find probability of two people with same birthday in group with size n
def same_birthday(iteration,n):
    same = 0
    for i in range(iteration):
        days =np.random.randint(1,365,n)#we made list of possible day for birthday
        temp = set(days)
        temp = list(temp)#we delete duplicated elements
        if len(days) != len(temp):#compare it with first list to find duplicate in first list
            same = same + 1
    return same/iteration

#first part
#for group of people we found probability
n = 50
iteration = 10000
p = same_birthday(iteration,n)
print('Probability when n =  ', n , ' is equal : ', p)

#second part
#now we use for to find probability for different size of groups
ps = []
iteration = 10000
for n in range(1,100):
    temp = same_birthday(iteration,n)
    ps.append(temp)
plt.plot(ps)
plt.xlabel('n')
plt.ylabel('Probability')
plt.title('Probability for different n')
plt.show()
