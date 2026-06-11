import pandas as pd
import numpy as np
from Bio import Align
import pickle

# Simple synergy function - hardcoded from your data
def get_synergies():
    """Return synergy pairs"""
    return [
        {"pair": "G1764A + G1899A", "synergy_effect": 55.7},
        {"pair": "PreS1 + PreS2", "synergy_effect": 52.6},
        {"pair": "G1896A + PreS1", "synergy_effect": 49.2},
        {"pair": "G1899A + C1653T", "synergy_effect": 45.8},
        {"pair": "A1762T + T1753V", "synergy_effect": 32.9}
    ]

def get_reference():
    """Load reference sequence"""
    with open('ref sequence.fasta', 'r') as f:
        lines = f.readlines()
        sequence = ''.join([line.strip() for line in lines if not line.startswith('>')])
    return sequence.upper()

def clean_sequence(raw_sequence):
    """Clean input sequence"""
    lines = raw_sequence.strip().split('\n')
    cleaned = ''.join([line.strip() for line in lines if not line.startswith('>')])
    cleaned = ''.join(cleaned.split())
    return cleaned.upper()

def align_sequences(query, reference):
    """Align using Biopython PairwiseAligner"""
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -2
    
    alignments = aligner.align(reference, query)
    best = alignments[0]
    return best[1]

def detect_genotype(sequence):
    """Detect HBV genotype"""
    genotype_scores = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
    
    # Genotype A
    if len(sequence) > 97 and sequence[0] == 'T':
        genotype_scores['A'] += 1
    if len(sequence) > 53 and sequence[52] == 'C':
        genotype_scores['A'] += 1
    if len(sequence) > 97 and sequence[96] == 'C':
        genotype_scores['A'] += 1
    
    # Genotype B
    if len(sequence) > 52 and sequence[51] == 'C':
        genotype_scores['B'] += 1
    if len(sequence) > 96 and sequence[95] == 'A':
        genotype_scores['B'] += 1
    if len(sequence) > 127 and sequence[126] == 'A':
        genotype_scores['B'] += 1
    if len(sequence) > 25 and sequence[24] == 'T':
        genotype_scores['B'] += 1
    if len(sequence) > 147 and sequence[146] == 'T':
        genotype_scores['B'] += 1
    
    # Genotype C
    if len(sequence) > 76 and sequence[75] == 'C':
        genotype_scores['C'] += 1
    if len(sequence) > 165 and sequence[164] == 'C':
        genotype_scores['C'] += 1
    if len(sequence) > 10 and sequence[9] == 'A':
        genotype_scores['C'] += 1
    if len(sequence) > 49 and sequence[48] == 'A':
        genotype_scores['C'] += 1
    
    # Genotype D
    if len(sequence) > 43 and sequence[42] == 'A':
        genotype_scores['D'] += 1
    if len(sequence) > 55 and sequence[54] == 'C':
        genotype_scores['D'] += 1
    if len(sequence) > 135 and sequence[134] == 'T':
        genotype_scores['D'] += 1
    if len(sequence) > 148 and sequence[147] == 'G':
        genotype_scores['D'] += 1
    
    # Genotype F
    if len(sequence) > 9 and sequence[8] == 'A':
        genotype_scores['F'] += 1
    if len(sequence) > 4 and sequence[3] == 'A':
        genotype_scores['F'] += 1
    if len(sequence) > 8 and sequence[7] == 'C':
        genotype_scores['F'] += 1
    if len(sequence) > 19 and sequence[18] == 'G':
        genotype_scores['F'] += 1
    
    best_genotype = max(genotype_scores, key=genotype_scores.get)
    return best_genotype if genotype_scores[best_genotype] > 0 else 'Unknown'

def extract_features(aligned_query, reference):
    """Extract mutation features"""
    features = {}
    additional = {}
    
    query_list = list(aligned_query)
    ref_list = list(reference)
    
    # Check A1762T (position 1762 in 1-indexed = 1761 in 0-indexed)
    if 1761 < len(query_list):
        features['A1762T'] = 1 if query_list[1761] == 'T' and ref_list[1761] == 'A' else 0
    
    # Check G1764A (position 1764 = 1763 in 0-indexed)
    if 1763 < len(query_list):
        features['G1764A'] = 1 if query_list[1763] == 'A' and ref_list[1763] == 'G' else 0
    
    # Check G1896A (position 1896 = 1895 in 0-indexed)
    if 1895 < len(query_list):
        features['G1896A'] = 1 if query_list[1895] == 'A' and ref_list[1895] == 'G' else 0
    
    # Check G1899A (position 1899 = 1898 in 0-indexed)
    if 1898 < len(query_list):
        features['G1899A'] = 1 if query_list[1898] == 'A' and ref_list[1898] == 'G' else 0
    
    # Check C1653T (position 1653 = 1652 in 0-indexed)
    if 1652 < len(query_list):
        features['C1653T'] = 1 if query_list[1652] == 'T' and ref_list[1652] == 'C' else 0
    
    # Check T1753V (position 1753 = 1752 in 0-indexed)
    if 1752 < len(query_list):
        features['T1753V'] = 1 if ref_list[1752] == 'T' and query_list[1752] != 'T' and query_list[1752] != '-' else 0
    
    # Check PreS1 deletion
    pres1_start, pres1_end = 2847, 3101
    if pres1_end < len(query_list):
        pres1_region = query_list[pres1_start:pres1_end+1]
        gap_count = pres1_region.count('-')
        features['PreS1'] = 1 if gap_count > len(pres1_region) * 0.5 else 0
    else:
        features['PreS1'] = 0
    
    # Check PreS2 deletion
    pres2_start, pres2_end = 3102, 3181
    if pres2_end < len(query_list):
        pres2_region = query_list[pres2_start:pres2_end+1]
        gap_count = pres2_region.count('-')
        features['PreS2'] = 1 if gap_count > len(pres2_region) * 0.5 else 0
    else:
        features['PreS2'] = 0
    
    # Additional mutations (not used in model)
    if 1612 < len(query_list):
        additional['G1613A'] = 1 if query_list[1612] == 'A' and ref_list[1612] == 'G' else 0
    else:
        additional['G1613A'] = 0
    
    if 1765 < len(query_list):
        additional['C1766T'] = 1 if query_list[1765] == 'T' and ref_list[1765] == 'C' else 0
    else:
        additional['C1766T'] = 0
    
    if 1767 < len(query_list):
        additional['T1768A'] = 1 if query_list[1767] == 'A' and ref_list[1767] == 'T' else 0
    else:
        additional['T1768A'] = 0
    
    return features, additional

def load_model():
    """Load the trained model"""
    with open('hcc_new model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def predict_hcc_risk(sequence):
    """Main prediction function"""
    
    model = load_model()
    reference = get_reference()
    cleaned_seq = clean_sequence(sequence)
    aligned = align_sequences(cleaned_seq, reference)
    genotype = detect_genotype(aligned)
    features, additional = extract_features(aligned, reference)
    
    # Prepare feature vector
    feature_order = ['A1762T', 'G1764A', 'G1896A', 'G1899A', 'C1653T', 'T1753V', 'PreS1', 'PreS2']
    feature_vector = np.array([[features[f] for f in feature_order]])
    
    # Predict
    risk_score = model.predict_proba(feature_vector)[0][1]
    risk_class = model.predict(feature_vector)[0]
    confidence = round(risk_score * 100, 1)
    
    # Prepare mutation details
    mutation_details = {mut: "Present" if features[mut] == 1 else "Absent" for mut in feature_order}
    additional_details = {mut: "Present" if additional[mut] == 1 else "Absent" for mut in additional}
    
    # Check for synergistic pairs
    synergies = get_synergies()
    detected_synergies = []
    for pair in synergies:
        pair_name = pair['pair']
        mut1, mut2 = pair_name.split(' + ')
        if features.get(mut1, 0) == 1 and features.get(mut2, 0) == 1:
            detected_synergies.append({
                'pair': pair_name,
                'extra_risk': pair['synergy_effect']
            })
    
    return {
        'risk': 'HIGH RISK' if risk_class == 1 else 'LOW RISK',
        'confidence': confidence,
        'genotype': genotype,
        'mutations': mutation_details,
        'additional_mutations': additional_details,
        'synergies': detected_synergies
    }
  
    
 
 
