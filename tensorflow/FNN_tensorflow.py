"""
FNN model - Tensorflow
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras import datasets, optimizers, losses, metrics

'''
クラスモデルの実装
'''
#構成 - 「隠れ層*3 - 出力層」
class DNN(Model):
  def __init__(self, hidden_dim, output_dim):
    super().__init__()

    self.l1 = Dense(hidden_dim, kernel_initializer = "he_normal") #重みの初期値 = Heの初期化
    self.b1 = BatchNormalization() #バッチ正規化
    self.a1 = Activation("relu") #Activation層
    self.d1 = Dropout(0.5) #Dropout

    self.l2 = Dense(hidden_dim, kernel_initializer = "he_normal")
    self.b2 = BatchNormalization()
    self.a2 = Activation("relu")
    self.d2 = Dropout(0.5)

    self.l3 = Dense(hidden_dim, kernel_initializer = "he_normal")
    self.b3 = BatchNormalization()
    self.a3 = Activation("relu")
    self.d3 = Dropout(0.5)

    self.l4 = Dense(output_dim, kernel_initializer = "he_normal", activation = "softmax")

    self.ls = [self.l1, self.b1, self.a1, self.d1,
              self.l2, self.b2, self.a2, self.d2,
              self.l3, self.b3, self.a3, self.d3,
              self.l4]

  def call(self, x):
    for layer in self.ls:
      x = layer(x)
    return x

#早期終了のクラス
class EarlyStopping:
  def __init__(self, patience = 0, verbose = 0):
    self._stop = 0
    self._loss = float("inf")
    self.patience = patience
    self.verbose = verbose

  def __call__(self, loss):
    if self._loss < loss:
      self._step += 1
      if self._step > self.patience:
        if self.verbose:
          print("early stopping")
        return True

    else:
      self._step = 0
      self.loss = loss

    return False

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
2.モデルの実装
'''
model = DNN(hidden_dim = 200, output_dim = 10)


'''
3.モデルの学習
'''
#事前準備
criation = losses.SparseCategoricalCrossentropy()
optimizer = optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = True)
train_loss = metrics.Mean()
train_acc = metrics.SparseCategoricalAccuracy()
val_loss = metrics.Mean()
val_acc = metrics.SparseCategoricalAccuracy()

def compute_loss(t, y):
  return criation(t, y)

#訓練データ処理用
def train_step(x, t):
  with tf.GradientTape() as tape:
    preds = model(x)
    loss = compute_loss(t, preds)

  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  train_loss(loss)
  train_acc(t, preds)

#検証データ処理用
def val_step(x, t):
  preds = model(x)
  loss = compute_loss(t, preds)
  val_loss(loss)
  val_acc(t, preds)

#学習開始
#ミニバッチ勾配降下法
epochs = 10
batch_size = 100
n_batches_train = x_train.shape[0]//batch_size
n_batches_val = x_val.shape[0]//batch_size

hist = {"loss" : [], "accuracy": [], "val_loss": [], "val_accuracy": []}

es = EarlyStopping(patience = 5, verbose = 1)

for epoch in range(epochs):
  x_, t_ = shuffle(x_train, t_train)

  for batch in range(n_batches_train):
    start = batch * batch_size
    end = start + batch_size
    train_step(x_[start:end], t_[start:end])

  for batch in range(n_batches_val):
    start = batch * batch_size
    end = start + batch_size
    val_step(x_val[start:end], t_val[start:end])

  hist["loss"].append(train_loss.result())
  hist["accuracy"].append(train_acc.result())
  hist["val_loss"].append(val_loss.result())
  hist["val_accuracy"].append(val_acc.result())

  print(f"epoch:{epoch+1}, loss:{train_loss.result():.3f}, acc:{train_acc.result():.3f}, val_loss:{val_loss.result():.3f}, val_acc:{val_acc.result():.3f}")

  if es(val_loss.result()):
    break

#4. モデルの評価
#データの可視化
loss = hist["loss"]
val_loss = hist["val_loss"]
fig = plt.figure()
plt.rc("font", family = "serif")
plt.plot(range(len(loss)), loss, color = "black", linewidth = 1)
plt.plot(range(len(val_loss)), val_loss, color = "red", linewidth = 1)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

#評価
test_loss = metrics.Mean()
test_acc = metrics.SparseCategoricalAccuracy()

def test_step(x, t):
  preds = model(x)
  loss = compute_loss(t, preds)
  test_loss(loss)
  test_acc(t, preds)

test_step(x_test, t_test)

print(f"test_loss: {test_loss.result():.3f}, test_acc: {test_acc.result():.3f}")