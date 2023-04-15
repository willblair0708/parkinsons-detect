# preprocess.py
import pandas as pd

def preprocess_data():
    clinical_data = pd.read_csv("../data/amp-parkinsons-disease-progression-prediction/train_clinical_data.csv")
    supplemental_data = pd.read_csv("../data/amp-parkinsons-disease-progression-prediction/supplemental_clinical_data.csv")
    peptide_data = pd.read_csv("../data/amp-parkinsons-disease-progression-prediction/train_peptides.csv")
    protein_data = pd.read_csv("../data/amp-parkinsons-disease-progression-prediction/train_proteins.csv")

    clinical_data = clinical_data.drop_duplicates()
    supplemental_data = supplemental_data.drop_duplicates()
    peptide_data = peptide_data.drop_duplicates()
    protein_data = protein_data.drop_duplicates()

    return clinical_data, supplemental_data, peptide_data, protein_data

