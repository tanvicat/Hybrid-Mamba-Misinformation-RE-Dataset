# Hybrid-Mamba-Misinformation-RE-Dataset
Dataset and code for the article: Beyond Transformers: A Hybrid Transformer-Mamba Approach for Misinformation Quantification in Multiclass Relation Extraction

# Hybrid Transformer–Mamba Dataset for Multiclass Misinformation Relation Extraction

This repository contains the datasets, preprocessing scripts, model implementations, and evaluation artifacts used in the paper:

**“Beyond Transformers: A Hybrid Transformer–Mamba Approach for Misinformation Quantification in Multiclass Relation Extraction” (2025)**

The repository is structured to support transparency, reproducibility, and reuse of the data and methods presented in the study.  
All data is organized by domain and label type (annotated, pseudo-labeled, and unlabeled), and the code is provided to reproduce the experiments end-to-end.

---

## 📁 Repository Structure
/data/

/annotated/
crime.csv
finance.csv
politics.csv
entertainment.csv

/pseudo_labeled/
crime_pseudo.csv
finance_pseudo.csv
politics_pseudo.csv
entertainment_pseudo.csv

/unannotated/
crime_unlabeled.csv
finance_unlabeled.csv
politics_unlabeled.csv
entertainment_unlabeled.csv


/code/
/Transformer/

/Mamba/

/Hybrid/


/models/
bert-base-uncased
microsoft/deberta-v3-base
vinai/bertweet-base
kornosk/polibertweet-political-twitter-roberta-mlm
yiyanghkust/finbert-pretrain

