from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import Model
from keras.layers import Input
from keras.callbacks import EarlyStopping

#1.데이터
x1 = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7],
            [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12]])
x2 = array([[20, 30, 40], [30, 40, 50], [40, 50, 60]])

y1 = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
y2 = array([50, 60, 70])

print("x1.shape : ", x1.shape)
print("x2.shape : ", x2.shape)
print("y1.shape : ", y1.shape)
print("y2.shape : ", y2.shape)


x1 = x[10, :]
x2 = x[10:, :]
y1 = y[:10]
y2 = y[10: ]

print(x1)
print(x2)
print(y1)
print(y2)

x1 = x1.reshape((x1.shape[0], x1.shape[1], 1))
x2 = x2.reshape((x2.shape[0], x2.shape[1], 1))

# 2. 모델구성

input1.add(LSTM(30, activation='relu', input_shape=(3,1)))
model1.add(Dense(10))(input1)
model1.add(Dense(3))(model1)


input2.add(LSTM(30, activation='relu', input_shape=(3,1))) 
model2.add(Dense(10))(input2)
model2.add(Dense(3))(model2)


from keras.layers.merge import concatenate
merge1 = concatenate([model1, model2])

output1 = Dense(30)(merge1)
output1 = Dense(10)(output1)
output11 = Dense(1)(output1)

output2 = Dense(30)(merge1)
output2 = Dense(10)(output2)
output22 = Dense(1)(output2)


model = Model(inputs = [input1, input2], outputs = [output1, output2])
model.summary()


# 3.실행
model.compile(optimizer='adam', loss='mse')

model.fit([x1, x1], [y1, y1], epochs=1000, batch_size=1)



'''
x1_input = array([25, 35, 45])
x1_input = x1_input.reshape((1,3,1))

x2_input = array([25, 35, 45])
x2_input = x2_input.reshape((1,3,1))

yhat = model.predict([x1_input, x2_input], batch_size=1, verbose=2)
print(yhat)


'''