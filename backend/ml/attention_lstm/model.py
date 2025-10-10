import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Layer, Input, LSTM, Dense, Dropout, Bidirectional)
from tensorflow.keras import backend as K

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="W",
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.u = self.add_weight(
            name="u", shape=(input_shape[-1], 1), initializer="glorot_uniform", trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        score_input = K.dot(x, self.W)
        score_input = K.tanh(score_input)
        score = K.dot(score_input, self.u)

        weights = K.softmax(score, axis=1)
        
        context_vector = K.sum(x * weights, axis=1)
        return context_vector

    def get_config(self):
        return super(AttentionLayer, self).get_config()
    
def AttentionLSTM(num_features, lstm_units=150, dropout_rate=0.3, seq_len=60):
    inputs = Input(shape=(seq_len, num_features))
    lstm_out = Bidirectional(
        LSTM(
            units=lstm_units,
            return_sequences=True,
            kernel_initializer="he_normal",
        )
    )(inputs)

    attention_output = AttentionLayer()(lstm_out)
    dropout_output = Dropout(rate=dropout_rate)(attention_output)
    outputs = Dense(1)(dropout_output)
    model = Model(inputs=inputs, outputs=outputs)
    
    return model