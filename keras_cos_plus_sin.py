from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Merge
from keras.utils.np_utils import to_categorical
from mpl_toolkits.mplot3d import Axes3D
import gc
from scipy.interpolate import griddata
from keras.callbacks import EarlyStopping


def count_error(arr1, arr2):
    arr_sub = arr1-arr2
    arr_abs = np.abs(arr_sub)
    arr_sum = np.sum(arr_abs)
    return arr_sum

model = Sequential()
model.add(Dense(32, input_dim = 2))
model.add(Activation("relu"))
model.add(Dense(output_dim=5, input_dim=1))
model.add(Activation("relu"))
model.add(Dense(output_dim=5, input_dim=1))
model.add(Activation("relu"))
model.add(Dense(output_dim=5, input_dim=1))
model.add(Activation("relu"))
model.add(Dense(output_dim=1))


early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)




model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

#Prepare training set
data1 = (np.random.random((10000))-.5) * 2 * np.pi
data2 = (np.random.random((10000))-.5) * 2 * np.pi
data = np.column_stack((data1, data2))
labels = np.sin(data[:,0]) + np.cos(data[:,1])
tr_exp = []
for i in labels:
    tr_exp.append([i])

#Prepare test set
test1 = np.linspace(-np.pi, np.pi, 10000)
test2 = np.linspace(-np.pi, np.pi, 10000)
test = np.column_stack((test1, test2))
test_temp_exp = np.sin(test[:,0]) + np.cos(test[:,1])
test_exp = []
for i in test_temp_exp:
    test_exp.append([i])




# train the model
model.fit(data, labels, nb_epoch=50, batch_size=32, callbacks=[early_stop])

#Test model
test_result = model.predict(test, batch_size=32, verbose=0)

tr_result = model.predict(data, batch_size=32, verbose=0)


#Visualize in 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

X = data1
Y = data2

#Training display
Z = labels
ax.scatter(X, Y, Z, c='r', marker='o')

#Test predict display
X = test1
Y = test2
Z = test_result
ax.scatter(X, Y, Z, c='b', marker='o')

#Test expected display
#Z = test_exp
#ax.scatter(X, Y, Z, c='g', marker='o')

difference_test = count_error(test_exp, test_result)
print("Difference(test prediction <--> test expected): ", difference_test)

difference_training = count_error(tr_exp, tr_result)
print("Difference(training prediction <-->training expected): ", difference_training)
#Surface
#X, Y = np.meshgrid(X,Y)
#line = ax.plot_surface(X, Y, Z, color='blue')

plt.show()
gc.collect()


