from gettext import npgettext
import sqlite3
from fastapi import FastAPI, UploadFile, File # pour créer une API Web à laquelle les utilisateurs peuvent envoyer des fichiers d'images pour traitement.
from PIL import Image #pour manipuler les images 
import io #Il est utilisé ici pour manipuler les données binaires des fichiers d'images
from kaggle.api.kaggle_api_extended import KaggleApi #Elle est utilisée ici pour télécharger un ensemble de données depuis Kaggle.
import requests
import speech_recognition as sr #reconnaître la parole à partir de l'audio en direct ou d'un fichier enregistré
import pyttsx3 #pyttsx3 est une bibliothèque de synthèse vocale en texte pour Python.
import cv2 # type:OpenCV signifie Open Source Computer Vision. Elle a été créée pour les applications de Computer Vision et accélérer leur déploiement. Elle prend en entrée de nombreuses entrées visuelles telles que des images et des vidéos. Elle permet par exemple de faire ce qu’on appelle de l’OCR (Optical character recognition) permettant de résoudre les problèmes de reconnaissance de caractère sur une image ou par exemple un PDF.
from bs4 import BeautifulSoup

app = FastAPI()

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    

    return {"message": "Image received and processed successfully"}


api = KaggleApi()
api.authenticate()  


dataset_name = "dataset-name"   #---------Téléchargement de datasets depuis Kaggle-------
api.dataset_download_files(dataset_name, unzip=True, path='./datasets/')


engine = pyttsx3.init()


def recognize_speech():
   
    recognizer = sr.Recognizer()

   
    with sr.Microphone() as source:
        print("Parlez maintenant...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    
    try:
        print("Reconnaissance vocale en cours...")
        text = recognizer.recognize_google(audio, language='fr-FR')
        print("Texte reconnu:", text)
        return text
    except sr.UnknownValueError:
        print("La reconnaissance vocale n'a pas pu comprendre l'audio.")
        return ""
    except sr.RequestError as e:
        print("Erreur lors de la récupération des résultats de la reconnaissance vocale ; {0}".format(e))
        return ""


def speak(text):
    print("Synthèse vocale:", text)
    engine.say(text)
    engine.runAndWait()

class IA_Medical:
    def __init__(self):
        self.base_de_connaissances = {
            'fièvre': ['grippe', 'infection virale', 'infection bactérienne'],
            'douleur thoracique': ['crise cardiaque', 'inflammation musculaire'],
            'maux de tête': ['migraine', 'tension', 'sinusite'],
            'toux': ['bronchite', 'pneumonie', 'asthme'],
            'fatigue': ['anémie', 'dépression', 'hypothyroïdie'],
            'douleur abdominale': ['appendicite', 'gastrite', 'calculs biliaires'],
            'douleur articulaire': ['arthrite', 'goutte', 'bursite'],
           
        }

    def diagnostiquer(self, symptomes):
        diagnostics = []
        for symptome in symptomes:
            if symptome in self.base_de_connaissances:
                diagnostics.extend(self.base_de_connaissances[symptome])
        return list(set(diagnostics))

if __name__ == "__main__":
    ia = IA_Medical()
    symptomes = input("Quels sont vos symptômes ? Séparez-les par des virgules : ").split(',')
    resultats = ia.diagnostiquer(symptomes)
    print("Les diagnostics possibles sont :", resultats)


    def diagnostiquer(self, symptomes):
        diagnostics = []
        for symptome in symptomes:
            if symptome in self.base_de_connaissances:
                diagnostics.extend(self.base_de_connaissances[symptome])
        return list(set(diagnostics))
    
    
    class BaseDeDonneesMedicaments:
           def __init__(self):
                self.medicaments = {
            'paracétamol': 'Traitement de la fièvre et des douleurs légères à modérées.',
            'ibuprofène': 'Traitement de la fièvre, des douleurs et de l\'inflammation.',
            'amoxicilline': 'Traitement des infections bactériennes telles que les infections des oreilles, des sinus, de la gorge, des voies urinaires et de la peau.',
            'simvastatine': 'Traitement du cholestérol élevé et de la prévention des maladies cardiovasculaires.',
            'lévothyroxine': 'Traitement de l\'hypothyroïdie (thyroïde inactive).',
             "aspirin": "Pain reliever, fever reducer, and anti-inflammatory",
             "ibuprofen": "Pain reliever, fever reducer, and anti-inflammatory",
             "loratadine": "Antihistamine",
             "montelukast": "Leukotriene receptor antagonist",
              "ipratropium bromide": "Anticholinergic medication",
               "albuterol": "Bronchodilator",
               "salmeterol": "Long-acting bronchodilator",
               "fluticasone": "Corticosteroid",
                "budesonide": "Corticosteroid"
        }

    def trouver_utilisation(self, medicament):
        if medicament.lower() in self.medicaments:
            return self.medicaments[medicament.lower()]
        else:
            return "Ce médicament n'est pas répertorié dans la base de données."

if __name__ == "__main__":
    bdd_medicaments = BaseDeDonneesMedicaments()
    while True:
        medicament = input("Entrez le nom du médicament (ou 'q' pour quitter) : ")
        if medicament.lower() == 'q':
            break
        utilisation = bdd_medicaments.trouver_utilisation(medicament)
        print(utilisation)


if __name__ == "__main__":
    ia = IA_Medical()
    while True:
        user_input = recognize_speech()

        
        if user_input:
            symptomes = user_input.split(',')
            resultats = ia.diagnostiquer(symptomes)
            print("Les diagnostics possibles sont :", resultats)


# Fonction pour extraire les URL des images d'une page HTML
def extract_image_urls(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img')
    img_urls = [img['src'] for img in img_tags]
    return img_urls


# Fonction pour l'analyse d'image
def analyze_image(image_data):
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1) # type: ignore
    # Analyse de l'image (ici, une simple détection de visage)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return len(faces)  # Retourne le nombre de visages détecté

# Fonction pour rechercher dans la base de données
def search_database(result):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM images WHERE result=?", (result,))
    rows = cursor.fetchall()
    conn.close()
    return rows