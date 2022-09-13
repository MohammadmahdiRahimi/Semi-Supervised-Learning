import matplotlib.pyplot as plt
import numpy as np
#you can change number of people  and iterations
n = 10000
s = 5000
#make exponential distribution in n size
expo = np.random.exponential(scale=0.5 ,  size=n) 
expo_group = []
#choose sample random 
for i in range(s):
    expo_group.append(np.random.choice(expo, size=50, replace=False))
expo_group = np.array(expo_group)
#find mean and var
expo_mean = np.mean(expo_group)
expo_std = np.std(expo_group)
print("Mean of exponential distribution = ", expo_mean)
print("Standard deviation of exponential distribution = ", expo_std)
#find mean of each group then plot
expo_group_mean = np.mean(expo_group, axis=1)
plt.hist(expo_group_mean, bins=50)
plt.title('exponential distribution')
plt.show()
#repeat same thing for binomial distribution
bin = np.random.binomial(20, 0.8, size=n)
bin_group = []
for i in range(s):
    bin_group.append(np.random.choice(bin, size=50, replace=False))
bin_group = np.array(bin_group)
bio_mean = np.mean(bin_group)
bio_std = np.std(bin_group)
print("Mean of binomial distribution =  ", bio_mean)
print("Standard deviation of binomial distribution =  ", bio_std)
bin_group_mean = np.mean(bin_group, axis=1)
plt.hist(bin_group_mean, bins=50)
plt.title(' binomial distribution')
plt.show()

