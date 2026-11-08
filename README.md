# Emotion-Controlled Text Generation — fine-tuning d'un petit modèle de langage

Projet personnel (novembre 2026) — Hamad Aldenawi, M2 Intelligence Artificielle, CERI / Avignon Université.

**Objectif :** fine-tuner un petit modèle de langage génératif **open source, exécutable hors ligne** (`distilgpt2`)
pour générer à la demande de courts textes exprimant une émotion donnée, puis **évaluer** le réalisme du contrôle,
la diversité, et appliquer une couche de **sécurité**.

## Données
[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) (anglais, 6 émotions : sadness, joy, love, anger, fear, surprise).
Sous-échantillon équilibré de 150 textes par émotion pour l'entraînement (CPU).

## Méthode
1. **Conditionnement par préfixe** : chaque exemple devient `joy: i feel ...` ; à la génération on donne `joy:`.
   (Un premier essai avec des tokens spéciaux nouveaux `<|joy|>` n'a pas appris le contrôle avec si peu
   d'entraînement : précision ≈ hasard. Le préfixe en texte, déjà compris par le modèle pré-entraîné, marche mieux.)
2. **Fine-tuning** causal LM (Hugging Face `Trainer`), 3 époques, lr 5e-4.
3. **Sécurité** (`safety.py`) : filtre de termes à risque et de grossièretés, les générations rejetées sont régénérées.
4. **Évaluation** (`evaluate.py`) : un classifieur juge (TF-IDF + régression logistique, précision 0.829 sur le test de
   dair-ai/emotion) vérifie si chaque texte généré exprime l'émotion demandée ; diversité mesurée par distinct-1/2.

## Résultats (20 générations par émotion, CPU) — `python evaluate.py`
| Émotion | sadness | joy | love | anger | fear | surprise | **moyenne** |
|---|---|---|---|---|---|---|---|
| Précision du contrôle | 0.80 | 0.80 | 0.40 | 0.25 | 0.80 | 0.15 | **0.53** |

Hasard = 0.167. distinct-1 = 0.190, distinct-2 = 0.619. Générations rejetées par le filtre de sécurité (risque + grossièretés) : 4.

**Limites :** très petit entraînement (900 exemples, CPU), les émotions proches (love/joy, surprise/fear) sont mal
séparées, et le juge lui-même se trompe ~17 % du temps. Les textes sont parfois répétitifs. Le filtre de sécurité
par mots-clés est volontairement simple ; un vrai système utiliserait un classifieur dédié.

## Démo
Phrases générées et résultats : https://hamadaldenawi.github.io/projets.html

## Pistes
Entraîner sur GPU avec plus de données (`--n_per_label 2000`), comparer avec des LLM open source plus grands (LoRA),
évaluation humaine du réalisme, et protocole éthique pour des sujets sensibles.

## Lancer
```bash
pip install -r requirements.txt
python train.py --n_per_label 150 --epochs 3
python generate.py --label joy --n 5
python evaluate.py
```
