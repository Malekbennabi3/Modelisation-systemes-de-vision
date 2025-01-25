# Modelisation-systemes-de-vision

Dans ce repositoire nous allons partager tous les travaux relatifs au module M2 VMI **Modelisation des systemes de vision** presenté par monsieur <ins>Camille KURTZ</ins>.

## Presentation de l'article "Towards a general-purpose foundation model for computational pathology" 

Une presentation PowerPoint de [l'article](https://www.nature.com/articles/s41591-024-02857-3) est disponible à l'adresse: [Presentation[English]](https://github.com/Malekbennabi3/Modelisation-systemes-de-vision/blob/main/Pr%C3%A9sentation%20Article.pdf) 


## Etat de l'Art
L'etat de l'art sur le thème "Apprentissage des biomarqueurs en oncologie à partir des lames histologiques de tissus en utilisant les techniques d'IA modernes" Consultable à [l'adresse suivante: ](https://github.com/Malekbennabi3/Modelisation-systemes-de-vision/blob/main/Etat_de_lart_BENNABI_ZHAO.pdf)

## Projet final
La presentation PowerPoint du deuxieme thème du projet final intitulé "Transformer-based unsupervised contrastive learning for histopathological image classification (CTransPath)" Consultable à [l'adresse suivante: ](https://github.com/Malekbennabi3/Modelisation-systemes-de-vision/blob/main/Modelisation%20systemes%20de%20vision.pdf)

## Rapport

# 1- Introduction:
- Contexte:
 Problème des annotations dans les images médicales: Les images histopathologiques
 nécessitent des annotations précises, mais les processus d'annotation sont coûteux et complexes.
 représentations utiles. 
 
 - Solution:
 Apprentissage auto-supervisé (SSL): Il exploite des données non annotées pour générer des
 Apprentissage contrastif sémantiquement pertinent (SRCL):  SRCL améliore les approches
 traditionnelles d'apprentissage contrastif en sélectionnant des paires positives avec des
 concepts visuels similaires dans un espace latent.
 CTransPath: Une architecture hybride combinant un réseau neuronal convolutif (CNN) pour
 les caractéristiques locales et un transformateur Swin pour les dépendances globales. 

# 2- Datasets:
- Dataset original:
  - PAIP (Pathology Artificial Intelligence Platform) WSI:  http://wisepaip.org/paip
  - TCGA (National Cancer Institute):https://portal.gdc.cancer.gov/
 ~15,000,000 images au total puis choix de 100 images de chaque WSI.
 après le pre-traitement un total de2,700,521 images histopathologiques non labélisées sont rassemblées.

- Dataset utilisé:
  - TCGA-COAD | The Cancer Genome Atlas: Cancer colorectal (Colon Adenocarcinoma)
 https://www.cancerimagingarchive.net/collection/tcga-coad/
  - TCGA-BRCA | The Cancer Genome Atlas:Cancer du sein (Breast Invasive Carcinoma)
https://www.cancerimagingarchive.net/collection/tcga-brca/
 ~52 000 images & 192 WSI (sous format png/svs)

# 3- Architecture:
Le nombre des Paramètres du modèle est de 27.5M et il prend des images RGB de taille 224 x 224 x 3
![Approche SRCL](https://github.com/Malekbennabi3/Modelisation-systemes-de-vision/blob/main/Capture%20d'%C3%A9cran%202025-01-18%20222113.png)

L'architecture principale utilisé est composé d'un CNN pour capturer les caracteristiques locales (Bordures et textures) et d'un Transformer pour le mecanisme d'attention global.
Le CNN utilisé est similaire aux reseaux Resnet avec 3 couches convolutives et le transformer utilisé est de type [Swin](https://arxiv.org/abs/2103.14030) avec 4 couches d'auto-attention dotées de fenêtres décalées (Shifted Windows).

![Architecture CTransPath](https://github.com/Malekbennabi3/Modelisation-systemes-de-vision/blob/main/Capture%20d'%C3%A9cran%202025-01-18%20224538.png)

# 4- Protocole experimental:
Dans ce projet nous avons voulu essayer deux approches differentes:

- La premiere approche consiste en l'utilisation des photos WSI directement en format .svs et de laisser le modèle lui meme extraire les caracteristiques discriminantes sous forme de patchs ordonnés
- Dans la seconde approche nous avons prealablement decoupé chaque WSI en 250 images .png de taille (512x512) et nous avons aussi melangé l'ordre des imagettes obtenues.
![Un aperçu des patchs](https://github.com/Malekbennabi3/Modelisation-systemes-de-vision/blob/main/Capture%20d'%C3%A9cran%202025-01-19%20212201.png)

# 5- Evaluation:

Pour la réalisation de ce projet nous avons realisé un finetuning sur la dernière couche du modèle pré-entrainé pour faire de la classification, le modèle pré-entrainé est disponible à [l'adresse suivante:](https://huggingface.co/jamesdolezal/CTransPath/blob/main/ctranspath.pth)
Concernant le split utilisé on a fait 70% pour l'entrainement et 15% pour l'entrainement et 15% pour la validation, les métriques utilisées sont:
- Pour la classification:
   - Accuracy
   - Cross Entropy Loss
- Pour l'extraction des feature:
   - Accuracy
   - Cosine Similarity [-1, 1]

# 6- Resultats obtenus:
- TCGA-COAD:- Classification Image:
 Accuracy:  78.3%
 Loss: 0.18-Feature extraction:
 Accuracy: 76%

- TCGA-BRCA:- Classification Image:
 Accuracy: 93%
 Loss: 0.04- Feature extraction:
 Accuracy: 82%

Le resultat de la classification sur un entrainement désequilibé (5200 classe1/ 20000 classe2) montre que le modèle est assez robuste (uniquement 4% d'ecart entre l'entrainement et la validation)
![resultat](https://github.com/Malekbennabi3/Modelisation-systemes-de-vision/blob/main/Capture%20d'%C3%A9cran%202025-01-20%20091146.png)

L'entrainement est relativement rapide puisque le modèle converge en moyenne à la 5 eme epoch.
![Loss](https://github.com/Malekbennabi3/Modelisation-systemes-de-vision/blob/main/Capture%20d'%C3%A9cran%202025-01-19%20225314.png)

# 7- Conclusion:
- Approche Hybride est moins sensible au contexte global (contrairement aux approches basées completement sur les transformers)
- Entraînement relativement rapide même sur un grand volume de données
- Bonne capacité de généralisation mêmes sur des données dispersées

# 8- Critiques et perspectives:
- Problèmes de compatibilité (Inférence, timm 0.5.4).
- Difficulté d'accès au dataset utilisé dans l'article.
- Quelques problèmes non corrigés du code(39 issues au [repository](https://github.com/Xiyue-Wang/TransPath/issues)).
- Difficulté d’utilisation du modèle sur de la segmentation et ROI
  
 Une nouvelle version CTransPath v2 est attendue prochainement avec des ameliorations d’au moins 5% par rapport à la version actuelle (Màj Avril 2024).
 
