# main.py
from preprocess import preprocess_data

def main():
    clinical_data, supplemental_data, peptide_data, protein_data = preprocess_data()
    
    # Add further analysis, model training, or evaluation code here.

if __name__ == "__main__":
    main()
