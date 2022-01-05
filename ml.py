from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from var import *
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

def loadDataset():
    global imagePaths, label_list, data, labels, lb
    for label in label_list:
        for imagePath in glob.glob(imagePaths+label+'/*.jpg'):
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (32, 32)).flatten()
            data.append(image)
            labels.append(label)

    np.array(data).shape

    data = np.array(data, dtype='float') / 255.0
    labels = np.array(labels)

    print(labels)

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    print(labels)

def splitDataset():
    global x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    print('Ukuran data train =', x_train.shape)
    print('Ukuran data test =', x_test.shape)

def ANNArchitecture(hyperParameter):
    global model

    learning_rate = hyperParameter()['learning_rate']

    model = Sequential()
    model.add(Dense(512, input_shape=(3072,), activation="relu"))
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(len(label_list), activation="softmax"))

    model.summary()

    opt_funct = SGD(learning_rate)

    model.compile(loss = 'categorical_crossentropy', optimizer = opt_funct, metrics = ['accuracy'])

def modelTrain(hyperParameter):
    global model

    max_epochs = hyperParameter()['max_epochs']

    H = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=max_epochs, batch_size=32)

    N = np.arange(0, max_epochs)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["accuracy"], label="train_acc")
    plt.plot(N, H.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch #")
    plt.legend()
    plt.show()
    model.save('prediksi_umur.h5')

def modelLoad():
    return load_model('prediksi_umur.h5')

def modelEvaluate():
    global model
    predictions = model.predict(x_test, batch_size=32)
    print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=label_list))

def modelTest(queryPath):
    query = cv2.imread(queryPath)
    output = query.copy()
    query = cv2.resize(query, (32, 32)).flatten()
    q = []
    q.append(query)
    q = np.array(q, dtype='float') / 255.0

    q_pred = model.predict(q)
    print("prediksi: ")
    print(q_pred)
    i = q_pred.argmax(axis=1)[0]
    label = lb.classes_[i]

    text = "{}: {:.2f}%".format(label, q_pred[0][i] * 100)
    cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow('Output', output)
    cv2.waitKey() 
    cv2.destroyWindow('Output')

def machineLearning():
    global model
    model = modelLoad()

    loadDataset()
    splitDataset()
    ANNArchitecture(hyperParameter)
    modelTrain(hyperParameter)
    modelEvaluate()

def byLoadModel():
    global model
    loadDataset()
    splitDataset()
    model = modelLoad()
    modelEvaluate()

def hyperParameter():
    return {
        "learning_rate": 0.0005,
        "max_epochs": 1000
    }

# machineLearning() # Jalankan jika dan hanya jika untuk data train
byLoadModel() # Model sudah jadi, tinggal diload

modelTest('dataset/test/01.jpg')
modelTest('dataset/test/03.jpg')
modelTest('dataset/test/04.jpg')
modelTest('dataset/test/12.jpg')
modelTest('dataset/test/17.jpg')
modelTest('dataset/test/20.jpg')
modelTest('dataset/test/27.jpg')
modelTest('dataset/test/30.jpg')
modelTest('dataset/test/31.jpg')
modelTest('dataset/test/35.jpg')
modelTest('dataset/test/37.jpg')
modelTest('dataset/test/66.jpg')
modelTest('dataset/test/67.jpg')
modelTest('dataset/test/68.jpg')
modelTest('dataset/test/69.jpg')
modelTest('dataset/test/71.jpg')
modelTest('dataset/test/73.jpg')
modelTest('dataset/test/75.jpg')