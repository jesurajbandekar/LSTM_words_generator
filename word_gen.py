import numpy as np
from keras.models import load_model
import sys
from lstm_model import model_gen



#dataset is also attached

with open('nietzsche.txt', encoding='utf-8') as f:
  text = f.read().lower()

chars = sorted(list(set(text))                            #total characters present
chars_indices = dict((c,i) for i,c in enumerate(chars))   #characters to indices
indices_chars = dict((i,c) for i,c in enumerate(chars))   #indices to characters

maxlen = 40  # Maximum number of characters in a sentence to be given as input
sentences = []
next_chars = []
step=3

for i in range(0, len(text) - maxlen, step):
  sentences.append(text[i:i+maxlen])
  next_chars.append(text[i+maxlen])

#Converting to one-hot representation of characters

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i,sentence in enumerate(sentences):
  for t,char in enumerate(sentence):
    x[i,t, chars_indices[char]] = 1
    
  y[i,chars_indices[next_chars[i]]] = 1

x = x.astype(int)
y = y.astype(int)


''' train model

model = model_gen()
  
    OR'''

#Load the pretrained Model
model = load_model('shakespeare.h5')


i = 40  #any random starting character from the dataset
for j in range(500):       #500 characters that will be printed
  x_sen= text[i+j:maxlen+i+j]

  x_t = np.zeros((1,maxlen,len(chars)), dtype=np.int)

  for t , ch in enumerate(x_sen):
    x_t[0,t, chars_indices[ch]] = 1

  y_t = model.predict(x_t)
  

  sys.stdout.write(indices_chars[int(np.argmax(y_t, axis=-1))])
  sys.stdout.flush()












