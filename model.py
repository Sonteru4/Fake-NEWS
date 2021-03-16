# importing the libraries
import numpy as np
import pandas as pd
import string
import re
import nltk
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split as tts

print("All the modules imported....")

train = pd.read_csv('train.csv')
print('Dataset loaded....')

# preprocessing the data

def cleaning(text):
    text = str(text)
    text = text.lower()
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    clean = re.compile('<.*?>')
    text = re.sub(clean,'',text)
    text = pattern.sub('', text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)        
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text) 
    text = re.sub(r"\'ll", " will", text)  
    text = re.sub(r"\'ve", " have", text)  
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"have't", "have not", text)

    text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    tokens = word_tokenize(text)
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    text = ' '.join(words)
    return text

print("Cleaning is taking place....")
train['text_new'] = train['title'] + train['author'] + train['text']
train['text_new'] = train['text_new'].map(cleaning)
print("Cleaning is done....")

print('Formulating the dependent and independent variable....')
train = train[['text_new','label']]
X = train['text_new'].values
y = train['label'].values

print("Splitting the data into train and test....")
xtrain,xtest,ytrain,ytest = tts(X,y,test_size=0.2,random_state=42,stratify=y)
print("Splitting done....")
print("Total data points in Train data....", ytrain.shape[0])
print("Total data points in the Test data....", ytest.shape[0])

# copnverting dependent variable into categorical variable
ytrainf = to_categorical(ytrain)
ytestf = to_categorical(ytest)

# converting to text to sequences
tokenizer=Tokenizer(20000,lower=True,oov_token='UNK')
tokenizer.fit_on_texts(xtrain)
xtrain = tokenizer.texts_to_sequences(xtrain)
xtest = tokenizer.texts_to_sequences(xtest)

xtrain = pad_sequences(xtrain,maxlen=300,padding='post')
xtest = pad_sequences(xtest,maxlen=300,padding='post')
print("Data preprocessing is over....")

# maing the model
print("Making the model....")
model = Sequential()
model.add(Embedding(20000,64,input_length=300))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64,return_sequences=True)))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Dense(2,activation="softmax"))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print("Model making done....")
print(model.summary())

# fitting it into the data
print("Running the model....")
hist = model.fit(xtrain,ytrainf,epochs=30,validation_data=(xtest,ytestf))

print("Saving the model into the disk....")
model.save('news.h5')
print("Model saved into the disk....")

# plotting the figures
print("Plotting the figures....")
plt.figure(figsize=(15,10))
plt.plot(hist.history['accuracy'],c='b',label='train')
plt.plot(hist.history['val_accuracy'],c='r',label='validation')
plt.title("Model Accuracy vs Epochs")
plt.xlabel("EPOCHS")
plt.ylabel("ACCURACY")
plt.legend(loc='lower right')
plt.savefig('accuracy.jpg')


plt.figure(figsize=(15,10))
plt.plot(hist.history['loss'],c='orange',label='train')
plt.plot(hist.history['val_loss'],c='g',label='validation')
plt.title("Model Loss vs Epochs")
plt.xlabel("EPOCHS")
plt.ylabel("LOSS")
plt.legend(loc='upper right')
plt.savefig('loss.jpg')
print("Figures saved in the disk....")

# testing the model
print("Testing the model....")
print("The result obtained is...\n")
model.evaluate(xtest,ytestf)

ypred = model.predict_classes(xtest)

print("Classification Report:\n",classification_report(ytest,ypred))
cf = confusion_matrix(ytest,ypred)
print("The confusion matrix is: \n",cf)

plt.figure(figsize=(15,10))
sns.heatmap(cf,annot=True,cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("confusion.jpg")



