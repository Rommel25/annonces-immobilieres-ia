from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from loguru import logger
import uvicorn
import numpy as np
import pandas as pd

app = FastAPI()

class PredictionData(BaseModel):
    surface: float


class PredictionDataAppartement(BaseModel):
    surface: float
    nbRooms: float
    nbWindows: float
    price: float
    city: str

model = LinearRegression()


modelSecond = LogisticRegression(max_iter=200)


modelThird = KNeighborsClassifier(n_neighbors=5)

label_encoder = LabelEncoder()

model_note = LinearRegression()


model_year = LinearRegression()


model_garage = LogisticRegression(max_iter=200)




# Variable pour vérifier si le modèle est entraîné
is_model_trained = False


label_encoder = LabelEncoder()



@app.post("/train")
async def train():
    global is_model_trained

    # Lire le fichier CSV
    df = pd.read_csv('appartements.csv')
    df['city'] = label_encoder.fit_transform(df['city'])

    # Prédiction du prix
    X = df[['surface']]  
    y = df['price']  
    model.fit(X, y)

    # Catégorie prix
    bins = [0, 150000, 250000, 400000, float('inf')]  
    labels = ['low', 'normal', 'high', 'scam'] 
    df['price_category'] = pd.cut(df['price'], bins=bins, labels=labels)

    X = df[['nbRooms', 'surface', 'nbWindows', 'price']]
    y = df['price_category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelSecond.fit(X_train, y_train)
    
    # Note 
    X_note = df[['city', 'surface', 'price']]  
    y_note = df['note']  
    model_note.fit(X_note, y_note)

    # Année
    X_year = df[['city']]  
    y_year = df['annee_construction']  
    model_year.fit(X_year, y_year)
    
    # Garage
    X_garage = df[['city', 'price']] 
    y_garage = df['garage'] 
    model_garage.fit(X_garage, y_garage)
    
    # Marquer le modèle comme entraîné
    is_model_trained = True
    return {"message": "Modèle entraîné avec succès."}




@app.post("/predict")
async def predict(data: PredictionData):

    global is_model_trained

    if not is_model_trained:
        raise HTTPException(
            status_code=400, detail="Le modèle n'est pas encore entraîné. Veuillez entraîner le modèle d'abord.")

    X_new = np.array([[data.surface]])

    predicted_price = model.predict(X_new)[0]

    logger.info(f"Prédiction faite pour surface: {data.surface}")
    logger.info(f"Prix prédit: {predicted_price}")

    return {"predicted_price": predicted_price}


# Prédiction de la note en fonction de la ville, surface et prix
@app.post("/predict-note")
async def predict_note(data: PredictionDataAppartement):
    
    global is_model_trained
    if not is_model_trained:
        raise HTTPException(status_code=400, detail="Les modèles ne sont pas encore entraînés.")

    city_encoded = label_encoder.transform([data.city])[0]

    X_new = np.array([[city_encoded, data.surface, data.price]])
    
    predicted_note = model_note.predict(X_new)[0]

    return {"predicted_note": predicted_note}

# Prédiction de l'année de construction de l'appartement
@app.post("/predict-year")
async def predict_year(data: PredictionDataAppartement):
    global is_model_trained
    if not is_model_trained:
        raise HTTPException(status_code=400, detail="Les modèles ne sont pas encore entraînés.")

    city_encoded = label_encoder.transform([data.city])[0]

    X_new = np.array([[city_encoded]])
    
    predicted_year = model_year.predict(X_new)[0]

    logger.info(f"Prédiction de l'année pour ville: {city_encoded} -> {predicted_year}")
    return {"predicted_year": predicted_year}

# Prédiction s'il y a un garage
@app.post("/predict-garage")
async def predict_garage(data: PredictionDataAppartement):
    
    global is_model_trained
    if not is_model_trained:
        raise HTTPException(status_code=400, detail="Les modèles ne sont pas encore entraînés.")

    city_encoded = label_encoder.transform([data.city])[0]

    X_new = np.array([[city_encoded, data.price]])
    
    predicted_garage = model_garage.predict_proba(X_new)[0][1]

    logger.info(f"Prédiction de garage pour ville: {city_encoded}, prix: {data.price} -> {predicted_garage}")
    return {"predicted_garage": float(predicted_garage)}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=5000)
