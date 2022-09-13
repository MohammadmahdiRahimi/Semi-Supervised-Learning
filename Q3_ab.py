import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
#this function split data to train with tag and trian without tag and test
def split_data(df):
    x_temp_train, x_test, y_temp_train, y_test = train_test_split(df, df.complication, test_size=0.25, random_state=0)
    x_temp_train = x_temp_train.assign(complication = y_temp_train)
    x_train, x_train_taged, y_train, y_train_taged = train_test_split(x_temp_train, x_temp_train.complication, test_size=1/75, random_state=0)
    return  x_train, x_train_taged, x_test, y_train, y_train_taged, y_test 
#load data
_path = 'Surgical.csv'
df = pd.read_csv(_path)
df = df.sample(frac=1, random_state=18).reset_index(drop=True)#shuffle data
x_train, x_train_taged, x_test, y_train, y_train_taged, y_test  = split_data(df)
plt.hist(y_train_taged)#plot histogram
plt.xticks([0,1],['No complication ', 'complication'])
plt.ylabel('Number')
plt.title('Number of  No complication and complication')
plt.show()
logisticRegr1 = LogisticRegression(max_iter = 600)#train regression logistic
logisticRegr1.fit(x_train_taged, y_train_taged)
y_test_pred = logisticRegr1.predict(x_test)#predict with model
#find F1 and accuracy and confusion matrix 
f1_test = f1_score(y_test, y_test_pred)
accuracy = accuracy_score(y_test, y_test_pred)
cm = pd.crosstab(y_test, y_test_pred)
print('F1 score = ',f1_test)
print('Accuracy = ', accuracy)
print(cm)