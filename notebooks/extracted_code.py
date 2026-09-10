import numpy as np
import pandas as pd
#Normalizing per class
#Bootstrapping to generate new data for each class
#given a class, the set of functions should be able to generate new data
#create a save data method, that converts the dataframe to a CSV file

#eventual goal: run original data and bootstrapped data to obtain further insight 
#into accuracy and prediction 
#obtains numpy array from csv files
c0_1 = np.transpose(np.genfromtxt('C0_1.csv', delimiter=','))
c1_1 = np.transpose(np.genfromtxt('C1_1.csv', delimiter=','))
c1_2 = np.transpose(np.genfromtxt('C1_2.csv', delimiter=','))
c2_2 = np.transpose(np.genfromtxt('C2_2.csv', delimiter=','))
c3_1 = np.transpose(np.genfromtxt('C3_1.csv', delimiter=','))
c3_2 = np.transpose(np.genfromtxt('C3_2.csv', delimiter=','))
c4_1 = np.transpose(np.genfromtxt('C4_1.csv', delimiter=','))
c4_2 = np.transpose(np.genfromtxt('C4_2.csv', delimiter=','))
phsh = np.transpose(np.genfromtxt('PhSH.csv', delimiter=','))
#converts data into dataframe
dfc0_1 = pd.DataFrame(data = c0_1[1:, :],
                  columns = c0_1[0, :])
dfc1_1 = pd.DataFrame(data = c1_1[1:, :],
                  columns = c1_1[0, :])
dfc1_2 = pd.DataFrame(data = c1_2[1:, :],
                  columns = c1_2[0, :])
dfc2_2 = pd.DataFrame(data = c2_2[1:, :],
                  columns = c2_2[0, :])
dfc3_1 = pd.DataFrame(data = c3_1[1:, :],
                  columns = c3_1[0, :])
dfc3_2 = pd.DataFrame(data = c3_2[1:, :],
                  columns = c3_2[0, :])
dfc4_1 = pd.DataFrame(data = c4_1[1:, :],
                  columns = c4_1[0, :])
dfc4_2 = pd.DataFrame(data = c4_2[1:, :],
                  columns = c4_2[0, :])
dfphsh = pd.DataFrame(data = phsh[1:, :],
                  columns = phsh[0, :])
#sample what dataframe looks like
#Top row is wavelength and every row after that is one spectra (one measurement)
dfc0_1
#normalizing data
def normalize(dataframe):
    means = np.mean(dataframe, axis=0)
    sigma = np.std(dataframe, axis=0)
    return (dataframe-means)/sigma

dfc0_1 = normalize(dfc0_1.iloc[:, :1014])
#dfc0_1['Type'] = 51 * ['c0']
#dfc0_1
dfc0_1.columns
#generates random indices 
#count is how many indices and max is highest index possible
def rand_nums(count, max):
    return np.random.randint(0, max, size=count)
#generates one random single data point across the 1015 wavelengths
def gen_datum(dataframe):
    datum = {}
    for column_name in dataframe.columns:
        indices = rand_nums(10, len(dataframe))
        column_sample = dataframe[column_name].values[indices]
        datum[column_name] = [np.mean(column_sample)]
    return pd.DataFrame(datum)
#generates x data points using gen_datum function
def gen_x_data(dataframe, x):
    all_data = pd.DataFrame({})
    for i in range(x):
        all_data = pd.concat([gen_datum(dataframe), all_data], ignore_index=True)
    return all_data

result = gen_datum(dfc0_1)
gen_x_data(dfc0_1, 5)
#generates each dataframe separately (5 rows each)
#and appends the type to the end of the row
c0_1_rand = gen_x_data(dfc0_1, 5)
c0_1_rand['Type'] = 'c0'
c1_1_rand = gen_x_data(dfc1_1, 5)
c1_1_rand['Type'] = 'c1'
c1_2_rand = gen_x_data(dfc1_2, 5)
c1_2_rand['Type'] = 'c1'
c2_2_rand = gen_x_data(dfc2_2, 5)
c2_2_rand['Type'] = 'c2'
c3_1_rand = gen_x_data(dfc3_1, 5)
c3_1_rand['Type'] = 'c3'
c3_2_rand = gen_x_data(dfc3_2, 5)
c3_2_rand['Type'] = 'c3'
c4_1_rand = gen_x_data(dfc4_1, 5)
c4_1_rand['Type'] = 'c4'
c4_2_rand = gen_x_data(dfc4_2, 5)
c4_2_rand['Type'] = 'c4'
phsh_rand = gen_x_data(dfphsh, 5)
phsh_rand['Type'] = 'phsh'

df_rand = pd.concat([c0_1_rand, c1_1_rand, c1_2_rand, c2_2_rand, c3_1_rand, c3_2_rand, c4_1_rand, c4_2_rand, phsh_rand])
#df_rand.drop(df_rand.columns[len(df_rand.columns)-1], axis=1, inplace=True)
df_rand
df_rand = df_rand.reset_index(drop=True)
df_norm_rand = normalize(df_rand.iloc[:, :1014])
df_norm_rand['Type'] = df_rand['Type']
df_norm_rand
#dfc0_1 = normalize(dfc0_1.iloc[:, :1014])
dfc0_1['Type'] = 51 * ['c0']
#dfc1_1 = normalize(dfc1_1.iloc[:, :1014])
dfc1_1['Type'] = 51 * ['c1']
#dfc1_2 = normalize(dfc1_2.iloc[:, :1014])
dfc1_2['Type'] = 51 * ['c1']
#dfc2_2 = normalize(dfc2_2.iloc[:, :1014])
dfc2_2['Type'] = 51 * ['c2']
#dfc3_1 = normalize(dfc3_1.iloc[:, :1014])
dfc3_1['Type'] = 51 * ['c3']
#dfc4_1 = normalize(dfc4_1.iloc[:, :1014])
dfc4_1['Type'] = 51 * ['c4']
#dfc4_2 = normalize(dfc4_2.iloc[:, :1014])
dfc4_2['Type'] = 51 * ['c4']
#dfphsh = normalize(dfphsh.iloc[:, :1014])
dfphsh['Type'] = 51 * ['phsh']
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df = pd.concat([dfc0_1, dfc1_1, dfc1_2, dfc2_2, dfc3_1, dfc3_2, dfc4_1, dfc4_2, dfphsh])
df = df.reset_index(drop=True)
df_normed = normalize(df.iloc[:, :1014])
df_normed['Type'] = df['Type']
#df_normed.drop(df_normed.columns[len(df_normed.columns)-1], axis=1, inplace=True)
df_normed.drop(df_normed.columns[0], axis=1, inplace=True)
df_normed = df_normed.sample(frac=1)
df_normed
one_hot = pd.get_dummies(df_normed['Type'])
df_normed = df_normed.drop('Type', axis = 1)
df_normed = df_normed.join(one_hot)
df_normed
data_array = df_normed.to_numpy().astype('float32')
data_array.shape
data_array = df_normed.to_numpy().astype('float32')

X = data_array[:, :1013]
y = data_array[:, 1013:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)
import matplotlib.pyplot as plt

df_normed = df_normed.sample(frac=1)
df_normed
X_train.shape, X_test.shape, y_train.shape, y_test.shape
labels = list(df_normed.columns)[1013:]
labels
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dropout, Dense, Conv1D, AveragePooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Input, BatchNormalization, Reshape, Flatten, MaxPooling1D
from tensorflow.keras.regularizers import l2

keras.backend.clear_session()

learning_rate = 2e-4
epochs = 200
batch_size = 128

#Convolutional Neural Network with 6 layers using Rectified Linear Unit 
#and then a final layer using softmax for classification
input = Input(shape=(1013, 1,))
x = Conv1D(8, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(input)
x = BatchNormalization()(x)
x = AveragePooling1D(3)(x)
x = Conv1D(16, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = AveragePooling1D(3)(x)
x = Conv1D(32, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = AveragePooling1D(3)(x)
x = Conv1D(32, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = AveragePooling1D(3)(x)
x = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = AveragePooling1D(3)(x)
x = Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.1)(x)
x = Flatten()(x)
type_out = Dense(len(labels), activation="softmax", name="type")(x)

#runs the model and fits it to the training set
model = keras.Model(inputs=input, outputs=[type_out], name="raman_model")
optim = tf.optimizers.legacy.Adam(learning_rate = learning_rate)

model.compile(loss = {'type': 'categorical_crossentropy'},
              optimizer = optim, metrics = {'type': 'accuracy'})

history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs = epochs, batch_size = batch_size, shuffle=True)
model.evaluate(X_test, y_test)
model.summary()
one_hot = pd.get_dummies(df_norm_rand['Type'])
df_norm_rand = df_norm_rand.drop('Type', axis = 1)
df_norm_rand = df_norm_rand.join(one_hot)
df_norm_rand
data_array_rand = df_norm_rand.to_numpy().astype('float32')
X_rand = data_array[:, :1013]
y_rand = data_array[:, 1013:]
y_pred = model.predict(X_rand)
#creates the classes from the prediction and the random data, because we have classifications not numerical values
from sklearn.metrics import accuracy_score
y_pred_classes = np.argmax(y_pred, axis=1)
y_rand_classes = np.argmax(y_rand, axis=1)
#prints accuracy
print(accuracy_score(y_rand_classes, y_pred_classes))
#Change in model accuracy as the model runs through the data
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#confusion matrix for the random generated data
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_rand_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
#heat map of model (that used the testing set for validation - not the random generated data)
pred = model.predict(X_test)
y_type_categorical = np.argmax(y_test, axis = 1)
pred_categorical = np.argmax(pred, axis = 1)
confm = confusion_matrix(y_type_categorical, pred_categorical, labels=None)
df_cm = pd.DataFrame(confm, index=labels, columns=labels)
ax = sn.heatmap(df_cm, cmap='Oranges', annot=True, fmt='d')
plt.show()

