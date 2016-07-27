from keras.models import Sequential

model = Sequential()
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback

model.add(Dense(output_dim=5, input_dim=1))
model.add(Activation("relu"))
model.add(Dense(output_dim=5, input_dim=1))
model.add(Activation("relu"))
model.add(Dense(output_dim=5, input_dim=1))
model.add(Activation("relu"))
model.add(Dense(output_dim=5, input_dim=1))
model.add(Activation("relu"))
model.add(Dense(output_dim=1))

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

class TrainingHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.valid_losses = []
        self.predictions = []
        self.i = 0
        self.save_every = 50

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.i += 1        
        if self.i % self.save_every == 0:        
            pred = model.predict(data)
            self.predictions.append(pred)
            
history = TrainingHistory()

#Depth doesn't seem to work so well
#Wide seems better for this task.
#Of course what we're doing is incredibly stupid

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-np.pi, np.pi, 100)

data = (np.random.random((100000))-.5) * 2 * np.pi
vals = np.sin(data)

val_data = (np.random.random((100000))-.5) * 2 * np.pi
val_vals = np.sin(data)

model.fit(data, 
          vals, 
          batch_size=32, 
          nb_epoch=100, 
          verbose=1, 
          validation_data=(val_data, val_vals), 
          callbacks=[early_stopping, history], 
          shuffle= True,
          show_accuracy=True,)


y = model.predict(x, batch_size=32, verbose=0)
plt.plot(x,y) 

plt.plot(x,np.sin(x))
plt.show()
#plt.legend()
plt.figure(figsize=(6, 3))
plt.plot(history.losses)
plt.ylabel('error')
plt.xlabel('iteration')
plt.title('training error')
plt.show()