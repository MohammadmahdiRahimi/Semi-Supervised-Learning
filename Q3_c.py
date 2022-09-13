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

iteration = 0#number of iteration
news = [1] #new data
num_news = len (news)
while num_news > 0:#this loop last untill we have no data to add
    logisticRegr = LogisticRegression(max_iter = 1000)#train regression logistic
    logisticRegr.fit(x_train_taged, y_train_taged)
    y_test_pred = logisticRegr.predict(x_test)#predict with model
    print('iteration = ', iteration)
    ps = logisticRegr.predict_proba(x_train)#predict with model
    preds = logisticRegr.predict(x_train)
    p0 = ps[:,0]
    p1 = ps[:,1]
    df_ps = pd.DataFrame([])
    df_ps['preds'] = preds
    df_ps['p0'] = p0
    df_ps['p1'] = p1
    df_ps.index = x_train.index
    news = pd.concat([df_ps.loc[df_ps['p0'] > 0.7], df_ps.loc[df_ps['p1'] > 0.7]], axis=0)#find possible data to tag
    num_news = len (news)
    print(num_news,' data added')
    x_train_taged = pd.concat([x_train_taged, x_train.loc[news.index]], axis=0)#add new data
    y_train_taged = pd.concat([y_train_taged, news.preds])      
    x_train = x_train.drop(index=news.index)#delete it from data without tag
    iteration = iteration +1


logisticRegr = LogisticRegression(max_iter = 1000)#train regression logistic
logisticRegr.fit(x_train_taged, y_train_taged)
y_test_pred = logisticRegr.predict(x_test)#predict with model
#find F1 and accuracy and confusion matrix 
f1_test = f1_score(y_test, y_test_pred)
accuracy = accuracy_score(y_test, y_test_pred)
cm = pd.crosstab(y_test, y_test_pred)
print('F1 score = ',f1_test)
print('Accuracy = ', accuracy)
print(cm)