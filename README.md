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
-Contexte:
 Problème des annotations dans les images médicales: Les images histopathologiques
 nécessitent des annotations précises, mais les processus d'annotation sont coûteux et complexes.
 représentations utiles. 
 -Solution:
 Apprentissage auto-supervisé (SSL): Il exploite des données non annotées pour générer des
 Apprentissage contrastif sémantiquement pertinent (SRCL):  SRCL améliore les approches
 traditionnelles d'apprentissage contrastif en sélectionnant des paires positives avec des
 concepts visuels similaires dans un espace latent.
 CTransPath: Une architecture hybride combinant un réseau neuronal convolutif (CNN) pour
 les caractéristiques locales et un transformateur Swin pour les dépendances globales. 

# 2- Datasets:
- Dataset original:
 PAIP (Pathology Artificial Intelligence Platform) WSI:  http://wisepaip.org/paip
 TCGA (National Cancer Institute):https://portal.gdc.cancer.gov/
 ~15,000,000 images au total puis choix de 100 images de chaque WSI.
 après le pre-traitement un total de2,700,521 images histopathologiques non labélisées sont rassemblées.

- Dataset utilisé:
  TCGA-COAD | The Cancer Genome Atlas: Cancer colorectal (Colon Adenocarcinoma)
 https://www.cancerimagingarchive.net/collection/tcga-coad/
 TCGA-BRCA | The Cancer Genome Atlas:Cancer du sein (Breast Invasive Carcinoma)
https://www.cancerimagingarchive.net/collection/tcga-brca/
 ~52 000 images & 192 WSI (sous format png/svs)
# 3-


 
