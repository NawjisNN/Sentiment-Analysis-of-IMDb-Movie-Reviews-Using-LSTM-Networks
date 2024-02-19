
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

num_words = 10000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)

maxlen = 250
train_data = pad_sequences(train_data, maxlen=maxlen)
test_data = pad_sequences(test_data, maxlen=maxlen)

model = Sequential([
    Embedding(num_words, 16, input_length=maxlen),
    LSTM(32, dropout=0.2, recurrent_dropout=0.2),  # Adding dropout to LSTM
    Dense(24, activation='relu', kernel_regularizer=l2(0.001)),  # L2 regularization
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Adding early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

model.fit(train_data, train_labels, epochs=10, validation_split=0.2, callbacks=[early_stopping])

results = model.evaluate(test_data, test_labels)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")
