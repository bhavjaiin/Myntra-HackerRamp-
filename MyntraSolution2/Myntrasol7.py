import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify
import numpy as np
import schedule
import threading
import time
import matplotlib.pyplot as plt

# Step 1: Data Collection and Preprocessing

def load_fashion_data():
    data = pd.read_csv('myntra_products_catalog.csv')
    return data

fashion_data = load_fashion_data()
if fashion_data.empty:
    raise ValueError("No data found. Ensure the data collection step is correct.")

# Adjusting column selection based on actual dataset structure
fashion_data = fashion_data[['ProductBrand', 'ProductName']]
fashion_data.columns = ['Trend', 'DIY_Technique']
fashion_data.to_csv('myntra_products_catalog.csv', index=False)
print("Fashion DataFrame:\n", fashion_data.head())  # Debug statement

# Step 2: Model Training

data = pd.read_csv('myntra_products_catalog.csv')
X = data['Trend']
y = data['DIY_Technique']

# Placeholder text processing
X_processed = pd.get_dummies(X)

print("Processed X:\n", X_processed.head())  # Debug statement
print("Labels y:\n", y.head())  # Debug statement

if X_processed.empty or len(y) == 0:
    raise ValueError("The dataset is empty. Ensure that the data collection and preprocessing steps are correct.")

# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(label_encoder.classes_), activation='softmax')  # Number of unique classes in y
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)
model.save('newfashion_ai_model.h5')

# Step 3: Interactive Platform

app = Flask(__name__)
model = tf.keras.models.load_model('newfashion_ai_model.h5')

@app.route('/generate_tutorial', methods=['POST'])
def generate_tutorial():
    user_input = request.json['user_input']
    # Placeholder text processing for user input
    user_input_processed = pd.get_dummies(pd.Series([user_input]), drop_first=True)
    user_input_processed = user_input_processed.reindex(columns=X_train.columns, fill_value=0)
    prediction = model.predict(np.array(user_input_processed))
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    tutorial = create_tutorial(predicted_label[0])
    return jsonify({'tutorial': tutorial})

def create_tutorial(prediction):
    # Placeholder tutorial generation
    return f"This is a tutorial based on the trend: {prediction}"

# Step 4: Trend Analysis and Suggestion

def update_model():
    new_fashion_data = load_fashion_data()
    new_fashion_data = new_fashion_data[['ProductBrand', 'ProductName']]
    new_fashion_data.columns = ['Trend', 'DIY_Technique']
    new_fashion_data.to_csv('myntra_products_catalog.csv', index=False)

    new_data = pd.read_csv('myntra_products_catalog.csv')
    new_X = new_data['Trend']
    new_y = new_data['DIY_Technique']
    new_X_processed = pd.get_dummies(new_X)

    new_X_processed = new_X_processed.reindex(columns=X_train.columns, fill_value=0)
    new_y_encoded = label_encoder.transform(new_y)
    
    model.fit(new_X_processed, new_y_encoded, epochs=10, validation_split=0.2)
    model.save('newfashion_ai_model.h5')

schedule.every().day.at("00:00").do(update_model)

# Step 5: Visualization of Training History

def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history)

if __name__ == '__main__':
    schedule_thread = threading.Thread(target=lambda: schedule.run_pending())
    schedule_thread.start()
    app.run(debug=True)
    
    while True:
        schedule.run_pending()
        time.sleep(1)
