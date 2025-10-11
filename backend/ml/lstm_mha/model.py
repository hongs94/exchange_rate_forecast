import tensorflow as tf

def build_model(look_back, num_features, lstm_units=150, dropout_rate=0.3, learning_rate=0.001, batch_size=None):
    inputs = tf.keras.Input(shape=(look_back, num_features), batch_size=batch_size)
    x = tf.keras.layers.LSTM(lstm_units, return_sequences=False)(inputs)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss="mse"
    )
    return model