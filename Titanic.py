import pandas as pd
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense
import keras


# male = 1, female = 0
# [0,1] not survived
# [1,0] survived


def read_datasets():
    dataframe = pd.read_csv("train.csv")
    dataframe = dataframe[['Survived', 'Pclass', 'Sex', 'Age']]
    train_label = []
    survived_col = dataframe['Survived']

    for _ in survived_col:
        if _ == 0:
            train_label.append([0, 1])
        else:
            train_label.append([1, 0])

    dataframe = dataframe.drop("Survived", 1)
    dataframe['Sex'].replace(['female', 'male'], [0, 1], inplace=True)
    dataframe['Age'].fillna(1, inplace=True)
    train_data = np.array(dataframe)
    train_label = np.array(train_label)

    dataframe = pd.read_csv("test.csv")
    dataframe = dataframe[['PassengerId', 'Pclass', 'Sex', 'Age']]
    test_id = dataframe['PassengerId']
    dataframe = dataframe.drop('PassengerId', 1)
    dataframe['Sex'].replace(['female', 'male'], [0, 1], inplace=True)
    dataframe['Age'].fillna(1, inplace=True)
    test_data = np.array(dataframe)
    test_id = np.array(test_id)
    return train_data, train_label, test_data, test_id

def train(train_data, train_label):

    tbCallBack = keras.callbacks.TensorBoard(log_dir='/tmp/tflearn_logs/', histogram_freq=0, write_graph=True,
                                             write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                             embeddings_metadata=None)
    model = Sequential()

    model.add(Dense(128, input_dim=3, activation='tanh'))

    model.add(Dense(64, activation='tanh'))

    model.add(Dense(32, activation='tanh'))

    model.add(Dense(16, activation='tanh'))

    model.add(Dense(8, activation='tanh'))

    model.add(Dense(2, activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(train_data, train_label, epochs=800, batch_size=60, callbacks=[tbCallBack], verbose=2)

    score = model.evaluate(train_data, train_label, batch_size=100)

    model.save('titanic.model')

    print(score)

def predict(test_data):
    model = load_model("titanic.model")
    model = model.predict(test_data)
    pred = []
    for m in model:
        pred.append(np.argmax(m))
    dataframe = pd.DataFrame(columns=['PassengerId', 'Survived'])
    dataframe['PassengerId'] = test_id
    dataframe['Survived'] = pred
    dataframe.to_csv("submit.csv",index=False)


train_data, train_label, test_data, test_id = read_datasets()
train(train_data, train_label)
predict(test_data)
