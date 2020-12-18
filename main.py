import os
import pandas as pd 
import numpy as np 
import pickle

#reading the csv files (test.csv is the input file) 
with open('AppCategory.pkl', 'rb') as f:
    app_category = pickle.load(f)

#READS THE PHONE ONE DAY DATA
phone = pd.read_csv('test2.csv')

# stores the cleaned input data
pred_var = []

#feature engineering

#1. combining app category and phone data 
arr = app_category["better_category"].to_numpy()
new = []
for x in range(0, len(arr)):
    if arr[x] in ['Food_&_Drinks', 'Music_&_Audio', 'Dating', 'Personal_Fitness', 'Sports']:
        new.append('lifestyle')
    elif arr[x] in ['Social_Networking']:
        new.append('social network')
    elif arr[x] in ['Email', 'Messages', 'Messaging', 'Instant_Messaging']:
        new.append('communication')
    elif arr[x] in ['Background_Process','Camera','Dialer','Phone','Phone_Assistant' ,'Phone_Optimization', 'Phone_Personalization' ,'Phone_Tools' ,'Auto_&_Vehicles', 'Book_Readers' ,'Calendar', 'Coupons' ,'Document_Editor' ,'Drawing' ,'Time_Tracker' ,'To_Do_List', 'To-Do_List' ,'Travel_Planning','Video_Players_&_Editors', 'Wearables', 'Weather', 'Remote_Administration', 'Security', 'Portfolio/Trading', 'Home_Automation', 'House_Search', 'Internet_Browser', 'Maps', 'Job_Search', 'Business_Management', 'Personal_Finance', 'Medical', 'Family_Planning', 'Mechanical_Turk', 'online_Shopping', 'Online_Shopping']:
        new.append('utility & tools')
    elif arr[x] in ['Game_Multiplayer', 'Game_Singleplayer', 'Entertainment','Streaming_Services']:
        new.append('games & entertainment')
    elif arr[x] in ['News','Education']:
        new.append('news & information outlet')
        
app_category['categories'] = new
app_category = app_category.rename(columns={"app_id": "application"})

combined1 = pd.merge(phone,app_category,on='application')
combined1.dropna()

#2 list of categorical index
lifestyle = np.where(combined1['categories'] == 'lifestyle')[0]
social = np.where(combined1['categories'] == 'social network')[0]
communication = np.where(combined1['categories'] == 'communication')[0]
utility = np.where(combined1['categories'] == 'utility & tools')[0]
games = np.where(combined1['categories'] == 'games & entertainment')[0]
news = np.where(combined1['categories'] == 'news & information outlet')[0]

order = [communication, games, lifestyle, news, social, utility]
combined1['duration'] = (combined1['endTimeMillis'] - combined1['startTimeMillis'])/1000

#3 frequency
frequency = []
for x in range(0, len(order)):
    frequency.append(len(order[x])) 
pred_var  += frequency

#4 categorical duration
cat_duration = []
for x in range(0, len(order)):
    cat_duration.append(pd.Series(combined1['duration'], index=order[x]).sum())
pred_var  += cat_duration

#5 duration proportion
dur_proportion = []
dur_proportion = np.array(cat_duration) / np.sum(cat_duration)
dur_proportion = dur_proportion.tolist()
pred_var  += dur_proportion

#6 frequency proportion
freq_proportion = []
freq_proportion = np.array(frequency) / np.sum(frequency)
freq_proportion = freq_proportion.tolist()
pred_var  += freq_proportion

#7 recency
recency = []
for x in range(0, len(order)):
    recency.append((pd.Series(pd.to_datetime([pd.Series(combined1['endTime'], index=order[x]).max()])).dt.round("D")[0] - pd.to_datetime([pd.Series(combined1['endTime'], index=order[x]).max()])).total_seconds()[0])
recency = [0 if x != x else x for x in recency]
pred_var  += recency

#8 monetary
monetary = []
for x in range(0, len(order)):
    if frequency[x] == 0:
        monetary.append(0)
    else:
        monetary.append(cat_duration[x] / frequency[x])
pred_var  += monetary

#9 earliest categorical data
earliest = []
for x in range(0, len(order)):
    earliest.append(-(pd.Series(pd.to_datetime([pd.Series(combined1['startTime'], index=order[x]).min()])).dt.normalize()[0] - pd.to_datetime([pd.Series(combined1['startTime'], index=order[x]).min()])).total_seconds()[0])
earliest = [0 if x != x else x for x in earliest]
pred_var  += earliest

#10 categorical variance
var_dur = []
for x in range(0, len(order)):
    var_dur.append(pd.Series(combined1['duration'], index=order[x]).var())
var_dur = [0 if x != x else x for x in var_dur]
pred_var  += var_dur

#11 categorical standard deviation
std_dur = []
for x in range(0, len(order)):
    std_dur.append(pd.Series(combined1['duration'], index=order[x]).std())
std_dur = [0 if x != x else x for x in std_dur]
pred_var  += std_dur

#12 categorical minimum duration
min_dur = []
for x in range(0, len(order)):
    min_dur.append(pd.Series(combined1['duration'], index=order[x]).min())
min_dur = [0 if x != x else x for x in min_dur]
pred_var  += min_dur

#13 categorical maximum duration
max_dur = []
for x in range(0, len(order)):
    max_dur.append(pd.Series(combined1['duration'], index=order[x]).max())
max_dur = [0 if x != x else x for x in max_dur]
pred_var  += max_dur

#prediction through models
with open('./models/2clusterModel.pkl', 'rb') as f:
    rfModel1 = pickle.load(f)

with open('./models/3clusterModel.pkl', 'rb') as f:
    rfModel2 = pickle.load(f)

#the result with two clusters (see value in the inference graphs)
result1 = rfModel1.predict(np.array(pred_var).reshape(1, -1))

#the result with three clusters (see value in the inference graphs)
result2 = rfModel2.predict(np.array(pred_var).reshape(1, -1))

print(result1)
print(result2)