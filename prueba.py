import sqlite3
from typing import List, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Conectar o crear una base de datos SQLite
def create_database():
    connection = sqlite3.connect("restaurants.db")
    cursor = connection.cursor()
    
    # Crear tabla de restaurantes si no existe
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS restaurants (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        cuisine TEXT NOT NULL,
        price_range TEXT NOT NULL,
        location TEXT NOT NULL,
        rating REAL NOT NULL
    )
    ''')
    
    # Insertar datos actualizados
    cursor.execute('''DELETE FROM restaurants''')  # Eliminar datos previos si existen
    cursor.executemany('''INSERT INTO restaurants (id, name, cuisine, price_range, location, rating) VALUES (?, ?, ?, ?, ?, ?)''', [
        (1, 'Mochomos Torreon', 'Mexicana', 'alto', 'Boulevard Independencia, Torreon', 5.0),
        (2, 'The Mezquite', 'Mexicana', 'bajo', 'Avenida Mariano Abasolo, Torreon', 5.0)
    ])
    
    connection.commit()
    connection.close()

# Crear la red neuronal
class IntentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(IntentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Preparar los datos para entrenar la red neuronal
def prepare_training_data():
    # Datos de ejemplo para entrenamiento
    examples = [
        ("Quiero algo mexicano y caro", "Mexicana", "alto"),
        ("Busco comida mexicana a precio medio", "Mexicana", "medio")
    ]

    texts = [example[0] for example in examples]
    cuisines = [example[1] for example in examples]
    prices = [example[2] for example in examples]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts).toarray()

    cuisine_mapping = {"Mexicana": 0}
    price_mapping = {"bajo": 0, "medio": 1, "alto": 2}

    y_cuisine = [cuisine_mapping[c] for c in cuisines]
    y_price = [price_mapping[p] for p in prices]

    return X, y_cuisine, y_price, vectorizer, cuisine_mapping, price_mapping

# Entrenar la red neuronal
def train_model():
    X, y_cuisine, y_price, vectorizer, cuisine_mapping, price_mapping = prepare_training_data()
    
    X_train, _, y_cuisine_train, _ = train_test_split(X, y_cuisine, test_size=0.2, random_state=42)
    X_train, _, y_price_train, _ = train_test_split(X, y_price, test_size=0.2, random_state=42)

    input_size = X_train.shape[1]
    hidden_size = 16
    output_size_cuisine = len(cuisine_mapping)
    output_size_price = len(price_mapping)

    model_cuisine = IntentClassifier(input_size, hidden_size, output_size_cuisine)
    model_price = IntentClassifier(input_size, hidden_size, output_size_price)

    criterion = nn.CrossEntropyLoss()
    optimizer_cuisine = optim.Adam(model_cuisine.parameters(), lr=0.01)
    optimizer_price = optim.Adam(model_price.parameters(), lr=0.01)

    epochs = 500
    for epoch in range(epochs):
        # Entrenamiento para tipo de comida
        optimizer_cuisine.zero_grad()
        outputs_cuisine = model_cuisine(torch.FloatTensor(X_train))
        loss_cuisine = criterion(outputs_cuisine, torch.LongTensor(y_cuisine_train))
        loss_cuisine.backward()
        optimizer_cuisine.step()

        # Entrenamiento para rango de precios
        optimizer_price.zero_grad()
        outputs_price = model_price(torch.FloatTensor(X_train))
        loss_price = criterion(outputs_price, torch.LongTensor(y_price_train))
        loss_price.backward()
        optimizer_price.step()

    return model_cuisine, model_price, vectorizer, cuisine_mapping, price_mapping

# Predecir usando la red neuronal
def predict_intent(model_cuisine, model_price, vectorizer, cuisine_mapping, price_mapping, user_input):
    X_input = vectorizer.transform([user_input]).toarray()
    cuisine_output = model_cuisine(torch.FloatTensor(X_input))
    price_output = model_price(torch.FloatTensor(X_input))

    predicted_cuisine = torch.argmax(cuisine_output, dim=1).item()
    predicted_price = torch.argmax(price_output, dim=1).item()

    cuisine = [k for k, v in cuisine_mapping.items() if v == predicted_cuisine][0]
    price = [k for k, v in price_mapping.items() if v == predicted_price][0]

    return {"cuisine": cuisine, "price_range": price}

# Buscar restaurantes en la base de datos
def recommend_restaurants(criteria: Dict[str, str]):
    connection = sqlite3.connect("restaurants.db")
    cursor = connection.cursor()

    query = "SELECT name, cuisine, price_range, location, rating FROM restaurants WHERE 1=1"
    params = []

    if criteria["cuisine"]:
        query += " AND cuisine = ?"
        params.append(criteria["cuisine"])

    if criteria["price_range"]:
        query += " AND price_range = ?"
        params.append(criteria["price_range"])

    cursor.execute(query, params)
    results = cursor.fetchall()
    connection.close()

    return [
        {"name": row[0], "cuisine": row[1], "price_range": row[2], "location": row[3], "rating": row[4]}
        for row in results
    ]

# Probar el sistema de recomendación
def main():
    create_database()
    model_cuisine, model_price, vectorizer, cuisine_mapping, price_mapping = train_model()

    print("¡Bienvenido al sistema de recomendación de restaurantes!")
    user_input = input("¿Qué estás buscando? ")
    
    criteria = predict_intent(model_cuisine, model_price, vectorizer, cuisine_mapping, price_mapping, user_input)
    recommendations = recommend_restaurants(criteria)

    if recommendations:
        print("\nTe recomendamos los siguientes restaurantes:")
        for r in recommendations:
            print(f"- {r['name']} ({r['cuisine']}, {r['price_range']} precio) en {r['location']} con calificación {r['rating']}")
    else:
        print("\nLo sentimos, no encontramos restaurantes que coincidan con tu búsqueda.")

if __name__ == "__main__":
    main()
