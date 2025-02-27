import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pickle

# Load the employee feedback dataset
data = pd.read_csv("employee_feedback.csv")

# Preprocess the data
texts = data["feedback"].values
labels = data["attrition_risk"].values

# Encode labels (low -> 0, medium -> 1, high -> 2)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
categorical_labels = to_categorical(encoded_labels)

# Tokenize the text data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to ensure uniform input size
max_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, categorical_labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=max_length),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(3, activation="softmax")  # 3 output classes: low, medium, high
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save("sentiment_model.h5")
print("Model saved as sentiment_model.h5")

# Save the tokenizer
with open("tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)
print("Tokenizer saved as tokenizer.pkl")
