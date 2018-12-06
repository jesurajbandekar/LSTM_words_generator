from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM



#The model is trained for 20 epochs  

def model_gen():
  model = Sequential()
  model.add(LSTM(256, input_shape=(maxlen,len(chars))))
  model.add(Dense(len(chars), activation='softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='adam')
  return model




