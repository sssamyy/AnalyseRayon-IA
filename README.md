# Détection de Produits et Analyse de Merchandising 🛒🤖

Ce projet repose sur une approche **d’intelligence artificielle** pour détecter les produits dans les rayons, analyser leur répartition et comparer l’espace occupé par la marque **Ramy** face à ses concurrents.

## 🧠 Objectifs

- Détecter et classifier les produits dans des images de rayons.
- Analyser le linéaire représenté par Ramy vs concurrents.
- Fournir des insights exploitables pour le **merchandising** et l’optimisation des ventes.

## 📂 Structure du Projet

📁 Product-Merchandising-AI
├── ai-model.py # Entraînement du modèle de détection
├── dataAugmentation.py # Augmentation des données
├── objectdetection.py # Détection d’objets via le modèle IA
├── scrap.py # Web scraping des produits concurrents
├── testModel.py # Tests du modèle sur des images nouvelles
└── README.md # Documentation

## 🔧 Fonctionnement

1. **Acquisition d’images** : Photos des rayons ou images via scraping.
2. **Détection d’objets** : Identification des produits (via TensorFlow, PyTorch ou autre modèle).
3. **Analyse des données détectées** : Calcul des zones occupées pour chaque produit.
4. **Comparaison & Insights** :
   - Quelle part de linéaire pour Ramy ?
   - Quels concurrents dominent l’espace ?
   - Recommandations visuelles et analytiques.

## 🚀 Installation

```bash
# Cloner le dépôt
git clone <repository-url>
cd Product-Merchandising-AI

# Installer les dépendances (exemple)
pip install -r requirements.txt
# Entraîner le modèle
python ai-model.py

# Scraper des images
python scrap.py

# Lancer la détection sur des images de test
python testModel.py
## ✅ Technologies

Python

OpenCV

TensorFlow / PyTorch

Selenium (pour scraping)

Matplotlib / Seaborn (visualisation)

## 🏁 Résultats attendus

Dashboard visuel des rayons détectés

Rapport comparatif Ramy vs concurrents

Fichier CSV/JSON exportable pour analyse métier
