from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow import keras
from tensorflow.keras import regularizers

# All sklearn Transforms must have the `transform` and `fit` methods    
# Autoencoder
class autocodificador(BaseEstimator, TransformerMixin):
    def __init__(self):
        model=keras.Sequential()
        # input 
        #model.add(keras.Input(shape=(16,)))
        
        # codificacion
        model.add(keras.layers.Dense(150, activation='tanh', activity_regularizer=regularizers.l1(10e-5)))
        model.add(keras.layers.Dense(70, activation='relu'))
        
        # decodificacion
        model.add(keras.layers.Dense(70, activation='tanh'))
        model.add(keras.layers.Dense(100, activation='tanh'))
        
        # output
        model.add(keras.layers.Dense(16,activation='relu'))
        
        # compile
        model.compile(optimizer="adadelta", loss="mse")
        
        # instancia
        self.autoencoder = model
        

    def fit(self, X, y):
        x_norm=X[y==0]
        self.autoencoder.fit(x_norm[3000:5000], x_norm[3000:5000], batch_size = 256, epochs = 50, shuffle = True, validation_split = 0.20)
        return self

    def transform(self, X1):
        # Aumentar la dimension de los datos para facilitar la deteccion de patrones
        hidden_representation = keras.Sequential()
        hidden_representation.add(self.autoencoder.layers[0])
        hidden_representation.add(self.autoencoder.layers[1])
        return hidden_representation.predict(X1)
    
    def fit_transform(self, X1, X2):
        self.fit(X1,X2)
        return self.transform(X1)
    
