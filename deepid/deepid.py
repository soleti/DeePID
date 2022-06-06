import numpy as np
import pandas as pd

from tqdm.keras import TqdmCallback

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, Concatenate, LSTM, Dropout, Reshape, Bidirectional

def build_global_df(arrays, global_vars):
    df = pd.DataFrame(arrays["detrkpid"][global_vars].to_numpy())
    df["redchisq"] = arrays["de"]["chisq"]/arrays["de"]["ndof"]
    df["muredchisq"] = arrays["dm"]["chisq"]/arrays["dm"]["ndof"]
    df["de_status"] = arrays["de"]["status"]
    df["t0"] = arrays["de"]["t0"]
    df["td"] = arrays["deent"]["td"]
    df["d0"] = arrays["deent"]["d0"]
    df["om"] = arrays["deent"]["om"]
    df["ue_status"] = arrays["ue"]["status"]
    df["mom"] = arrays["deent"]["mom"]
    df["trkpid"] = arrays["detrkpid"]["mvaout"]
    df["gen"] = arrays["demcgen"]["gen"]
    df["pdg"] = arrays["demc"]["pdg"]

    return df


def prepare_dataset(arrays, scaler, global_vars, new_vars, n_bins=101, maxlen=50, fit=False):  
    df_global = build_global_df(arrays, global_vars)

    mask = arrays['detsh._dx']>0
    dedx = (arrays['detsh._edep'][mask]/arrays['detsh._dx'][mask]).to_list()
    dedx_digitized = [np.digitize(e, bins=np.linspace(-0.005,0.02,n_bins)) for e in dedx]
    dedx_padded = sequence.pad_sequences(dedx_digitized, dtype='int32', maxlen=maxlen, padding='post', truncating='post')
    
    dedx_median = [np.median(e) for e in dedx]
    df_global["dedx"] = dedx_median

    if fit:
        global_array = scaler.fit_transform(df_global[new_vars].to_numpy())
    else:
        global_array = scaler.transform(df_global[new_vars].to_numpy())
        
    residual = arrays['detsh._resid'][mask]
    residual_digitized = [np.digitize(r, bins=np.linspace(-10,10,n_bins)) for r in residual]
    residual_padded = sequence.pad_sequences(residual_digitized, dtype='int32', maxlen=maxlen, padding='post', truncating='post')

    plane = arrays['detsh._plane'][mask]+1
    plane_padded = sequence.pad_sequences(plane, dtype='int32', maxlen=maxlen, padding='post', truncating='post')

    return df_global, global_array, dedx_padded, residual_padded, plane_padded

def build_lstm_network(max_features, embedding_size=64, maxlen=50):
    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(embedding_size*2)))
    model.add(Dropout(0.2))
    model.add(Dense(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def build_sample(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    
    return x_train, x_test, x_valid, y_train, y_test, y_valid

def train_model(model, x, y, x_valid, y_valid, epochs=20):
    history = model.fit(x, y,
                        epochs=epochs, 
                        verbose=0,
                        callbacks=[TqdmCallback(verbose=1)],
                        validation_data=(x_valid, y_valid)) 
    
    return history

def run_network(variable, target, embedding_size=64, maxlen=50, epochs=20):
    max_features = np.max(variable)+1
    
    model = build_lstm_network(max_features, embedding_size, maxlen)
    x_train, x_test, x_valid, y_train, y_test, y_valid = build_sample(variable, target)
    samples =  x_train, x_test, x_valid, y_train, y_test, y_valid
    history = train_model(model, x_train, y_train, x_valid, y_valid, epochs=epochs)
    
    prediction = model.predict(x_test).ravel()
    fpr, tpr, th = roc_curve(y_test,  prediction)
    auc = roc_auc_score(y_test, prediction)
    
    return samples, model, history, fpr, tpr, auc, prediction


def model_global(n_variables):
    model_global = Sequential()
    model_global.add(Dense(n_variables*8, activation='relu', input_shape=(n_variables,)))
    model_global.add(Dropout(0.2))
    model_global.add(Dense(n_variables*8, activation='relu'))
    model_global.add(Dropout(0.2))
    model_global.add(Dense(1, activation='sigmoid'))
    model_global.compile(loss='binary_crossentropy',
                         metrics='accuracy',
                         optimizer='adam')
    
    return model_global


def model_deepid(n_bins, n_variables, embedding_size, maxlen):
    model_multi = Sequential()
    model_multi.add(Embedding(n_bins+1, embedding_size, input_shape=(3,maxlen)))
    model_multi.add(Reshape([3,maxlen*embedding_size]))
    model_multi.add(Dropout(0.2))
    model_multi.add(Bidirectional(LSTM(embedding_size*2)))
    model_multi.add(Dropout(0.2))
    model_multi.add(Dense(1, activation='sigmoid'))
    model_multi.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    this_global = model_global(n_variables)
    
    mergedOutput = Concatenate()([model_multi.output, this_global.output])
    out = Dense(4*4, activation='relu', input_shape=(4,))(mergedOutput)
    out = Dropout(0.2)(out)
    out = Dense(1, activation='sigmoid')(out)

    model_deepid = Model(
        [model_multi.input, this_global.input],
        out
    ) 
    model_deepid.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])
    
    return model_deepid