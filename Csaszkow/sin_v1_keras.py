from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt

#Callbacks for metrics
class TrainingHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.valid_losses = []
        self.accs = []
        self.valid_accs = []
        self.epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 1 == 0:
            self.losses.append(logs.get('loss'))
            self.valid_losses.append(logs.get('val_loss'))
            self.accs.append(logs.get('acc'))
            self.valid_accs.append(logs.get('val_acc'))
            self.epoch += 1
            
            
early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
history = TrainingHistory()

#Create and set model
model = Sequential()
model.add(Dense(output_dim=5, input_dim=1))
model.add(Activation("relu"))
model.add(Dense(output_dim=5, input_dim=1))
model.add(Activation("relu"))
model.add(Dense(output_dim=5, input_dim=1))
model.add(Activation("relu"))
model.add(Dense(output_dim=5, input_dim=1))
model.add(Activation("relu"))
model.add(Dense(output_dim=1))
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])


#Prepare training data
tr_set = (np.random.random((10000))-.5) * 2 * np.pi
tr_vals = np.sin(tr_set)


#Prepare validation data
val_set = (np.random.random((10000))-.5) * 2 * np.pi
val_vals = np.sin(val_set)

#Prepare test data
tst_set = np.linspace(-np.pi, np.pi, 100)
test_vals = np.sin(tst_set)

#Train model
model.fit(tr_set, 
          tr_vals, 
          batch_size=32, 
          nb_epoch=100, 
          verbose=1, 
          validation_data=(val_set, val_vals), 
          callbacks=[early_stopping, history], 
          shuffle= True)

#Predict result
pred_res = model.predict(tst_set, batch_size=32, verbose=0)

#Viszualize prediction <--> expectation
plt.figure(figsize=(10, 4))
plt.title('Prediction')
plt.plot(tst_set,pred_res) 
plt.plot(tst_set,test_vals) 
plt.show()

#Viszualize losses
plt.figure(figsize=(10, 4))
plt.title('Loss curves')
plt.plot(np.arange(history.epoch), history.losses) 
plt.plot(np.arange(history.epoch), history.valid_losses) 
plt.show()

#Viszualize accuracies
plt.figure(figsize=(10, 4))
plt.title('Accuracy curves')
plt.plot(np.arange(history.epoch), history.accs) 
plt.plot(np.arange(history.epoch), history.valid_accs) 
plt.show()


#plt.ylabel('error')
#plt.xlabel('iteration')
#plt.title('training error')
