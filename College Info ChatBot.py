#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random


# In[2]:


each_words=[]
batches=[]
files=[]
ignore=["?","!"]


# In[3]:


data_file=open("D:/deva/Intents-29-03-2021").read()
data=json.loads(data_file)
data


# In[4]:


for content in data["intents"]:
    for pattern in content["patterns"]:
        data1=nltk.word_tokenize(pattern)
        each_words.extend(data1)
        files.append((data1,content["tag"]))
        if content["tag"] not in batches:
            batches.append(content["tag"])


# In[5]:


each_words=[lemmatizer.lemmatize(data1.lower()) for data1 in each_words if data1 not in ignore]
each_words=sorted(list(each_words))
batches=sorted(list(batches))
print(len(files),"files")
print(len(batches),"batches",batches)
print(len(each_words),"each_words",each_words)
pickle.dump(each_words,open("each_words.pkl","wb"))
pickle.dump(batches,open("batches.pkl","wb"))


# In[6]:


training_data=[]
empty_box=[0]*len(batches)
for doc in files:
    bag=[]
    pattern_words=doc[0]
    pattern_words=[lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for data1 in each_words:
        bag.append(1) if data1 in pattern_words else bag.append(0)
    row=list(empty_box)
    row[batches.index(doc[1])]=1
    training_data.append([bag,row])
random.shuffle(training_data)
training_data=np.array(training_data)
x=list(training_data[:,0])
y=list(training_data[:,1])


# In[7]:


machine=Sequential()
machine.add(Dense(488,input_shape=(len(x[0]),),activation="relu"))
machine.add(Dropout(0.5))
machine.add(Dense(244,activation="relu"))
machine.add(Dropout(0.5))
machine.add(Dense(len(y[0]),activation="softmax"))
gradiant=SGD(momentum=0.09,nesterov=True,decay=1e-6,lr=0.01)
machine.compile(loss="categorical_crossentropy",metrics=["accuracy"],optimizer=gradiant)
chat=machine.fit(np.array(x),np.array(y),epochs=250,batch_size=5,verbose=1)
machine.save("Chatbot.h5",chat)


# In[8]:


from keras.models import load_model
machine=load_model("Chatbot.h5")
each_words=pickle.load(open("each_words.pkl","rb"))
batches=pickle.load(open("batches.pkl","rb"))


# In[9]:


def sent(sentence):
    sent_words=nltk.word_tokenize(sentence)
    sent_words=[lemmatizer.lemmatize(word.lower()) for word in sent_words]
    return sent_words

def bow(sentence,each_words,show_details=True):
    sent_words=sent(sentence)
    bag=[0]*len(each_words)
    for s in sent_words:
        for i,data1 in enumerate(each_words):
            if data1==s:
                bag[i]=1
                if show_details:
                    print("found in bag:%s" %w)
    return(np.array(bag))

def prediction(sentence,machine):
    p=bow(sentence,each_words,show_details=False)
    res=machine.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": batches[r[0]], "probability": str(r[1])})
    return return_list

def reply(ints,list_intents):
    a=ints[0]["intent"]
    b=list_intents["intents"]
    for i in b:
        if(i["tag"]==a):
            result=random.choice(i["responses"])
            break
    return result  

def chatbot(messege):
    a=prediction(messege,machine)
    b=reply(a,data)
    return b


# In[10]:


chatbot("Who created you?")

