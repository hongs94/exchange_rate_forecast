import tensorflow as tf

class Attention(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(input_shape[-1], activation="tanh")
        self.dense2 = tf.keras.layers.Dense(1)
        super().build(input_shape)

    def call(self, x):
        score = self.dense2(self.dense1(x))
        weights = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(x * weights, axis=1)

def build_model(look_back, num_features, lstm_units=150, dropout_rate=0.3, learning_rate=0.001):
    inputs = tf.keras.Input(shape=(look_back, num_features))
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units, return_sequences=True)
    )(inputs)
    x = Attention()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse")
    return model
