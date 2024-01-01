"""
FNN model - keras
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras import datasets, optimizers
from tensorflow.keras.callbacks import EarlyStopping

'''
1.データの準備
'''
#MNISTと呼ばれる28*28ピクセルの画像データを用いる
mnist = datasets.mnist
(x_train, t_train), (x_test, t_test) = mnist.load_data()

#(28, 28)の次元だとモデルが対応できないため28 * 28である784をピクセルの最大値である255で割る 
x_train = (x_train.reshape(-1, 784)/255).astype(np.float32) 
x_test = (x_test.reshape(-1, 784)/255).astype(np.float32)

#訓練データ:検証データ = 8:2に分割
x_train, x_val, t_train, t_val = train_test_split(x_train, t_train, test_size = 0.2) 


'''
2. クラスの実装
'''
model = Sequential()

#隠れ層
model.add(Dense(200, kernel_initializer = "he_normal")) #重みの初期値 = Heの初期化
model.add(BatchNormalization()) #バッチ正規化
model.add(Activation("relu")) #Activation層
model.add(Dropout(0.5)) #Dropout

model.add(Dense(200, kernel_initializer = "he_normal"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(200, kernel_initializer = "he_normal"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))

#出力層
model.add(Dense(10, kernel_initializer = "he_normal", activation = "softmax"))


'''
3. モデルの学習
'''
optimizer = optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = True) #重みの更新式 = Adam

model.compile(optimizer = optimizer, loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

es = EarlyStopping(monitor = "val_loss", patience = 5, verbose = 1)

hist = model.fit(x_train, t_train, epochs = 10, batch_size = 100, verbose = 2, validation_data = (x_val, t_val), callbacks = [es])


'''
#4. モデルの評価
'''
#データの可視化
loss = hist.history["loss"]
val_loss = hist.history["val_loss"]
fig = plt.figure()
plt.rc("font", family = "serif")
plt.plot(range(len(loss)), loss, color = "black", linewidth = 1)
plt.plot(range(len(val_loss)), val_loss, color = "red", linewidth = 1)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

#評価
loss, acc = model.evaluate(x_test, t_test, verbose = 0)

print(f"test_loss: {loss:.3f}, test_acc: {acc:.3f}")