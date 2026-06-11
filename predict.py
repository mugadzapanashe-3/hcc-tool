import pandas as pd
import numpy as np
from Bio import Align
import re
import os
import json

# Simple synergy function without complex imports
def get_synergies():
    """Return pre-computed synergy pairs from cache file"""
    try:
        if os.path.exists('synergy_cache.json'):
            with open('synergy_cache.json', 'r') as f:
                return json.load(f)
    except:
        pass
    return []

# Get reference sequence
def get_reference():
    """Load the reference sequence from fasta file"""
    with open('ref sequence.fasta', 'r') as f:
        lines = f.readlines()
        sequence = ''.join([line.strip() for line in lines if not line.startswith('>')])
    return sequence.upper()

# Clean input sequence
def clean_sequence(raw_sequence):
    """Remove FASTA headers, line breaks, and spaces"""
    lines = raw_sequence.strip().split('\n')
    cleaned = ''.join([line.strip() for line in lines if not line.startswith('>')])
    cleaned = ''.join(cleaned.split())
    return cleaned.upper()

# Align sequences
def align_sequences(query, reference):
    """Align query sequence to reference using Biopython PairwiseAligner"""
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -2
    
    alignments = aligner.align(reference, query)
    best = alignments[0]
    
    # Get the aligned query sequence
    aligned_query = best[1]
    return aligned_query

# Detect genotype
def detect_genotype(sequence):
    """Detect HBV genotype using signature positions"""
    genotype_scores = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
    
    # Genotype A signatures
    if len(sequence) > 97 and sequence[0] == 'T':  # position 1
        genotype_scores['A'] += 1
    if len(sequence) > 53 and sequence[52] == 'C':  # position 53
        genotype_scores['A'] += 1
    if len(sequence) > 97 and sequence[96] == 'C':  # position 97
        genotype_scores['A'] += 1
    
    # Genotype B signatures
    if len(sequence) > 52 and sequence[51] == 'C':  # position 52
        genotype_scores['B'] += 1
    if len(sequence) > 96 and sequence[95] == 'A':  # position 96
        genotype_scores['B'] += 1
    if len(sequence) > 127 and sequence[126] == 'A':  # position 127
        genotype_scores['B'] += 1
    if len(sequence) > 25 and sequence[24] == 'T':  # position 25
        genotype_scores['B'] += 1
    if len(sequence) > 147 and sequence[146] == 'T':  # position 147
        genotype_scores['B'] += 1
    
    # Genotype C signatures
    if len(sequence) > 76 and sequence[75] == 'C':  # position 76
        genotype_scores['C'] += 1
    if len(sequence) > 165 and sequence[164] == 'C':  # position 165
        genotype_scores['C'] += 1
    if len(sequence) > 10 and sequence[9] == 'A':  # position 10
        genotype_scores['C'] += 1
    if len(sequence) > 49 and sequence[48] == 'A':  # position 49
        genotype_scores['C'] += 1
    
    # Genotype D signatures
    if len(sequence) > 43 and sequence[42] == 'A':  # position 43
        genotype_scores['D'] += 1
    if len(sequence) > 55 and sequence[54] == 'C':  # position 55
        genotype_scores['D'] += 1
    if len(sequence) > 135 and sequence[134] == 'T':  # position 135
        genotype_scores['D'] += 1
    if len(sequence) > 148 and sequence[147] == 'G':  # position 148
        genotype_scores['D'] += 1
    
    # Genotype F signatures
    if len(sequence) > 9 and sequence[8] == 'A':  # position 9
        genotype_scores['F'] += 1
    if len(sequence) > 4 and sequence[3] == 'A':  # position 4
        genotype_scores['F'] += 1
    if len(sequence) > 8 and sequence[7] == 'C':  # position 8
        genotype_scores['F'] += 1
    if len(sequence) > 19 and sequence[18] == 'G':  # position 19
        genotype_scores['F'] += 1
    
    best_genotype = max(genotype_scores, key=genotype_scores.get)
    best_score = genotype_scores[best_genotype]
    
    if best_score == 0:
        return 'Unknown'
    return best_genotype

# Extract mutations
def extract_features(aligned_query, reference):
    """Extract all 8 mutation features from aligned sequence"""
    features = {}
    additional = {}
    
    # Convert to lists for easier indexing
    query_list = list(aligned_query)
    ref_list = list(reference)
    
    # Position mappings (0-indexed)
    positions = {
        'A1762T': 1761,  # position 1762 in 1-indexed
        'G1764A': 1763,  # position 1764 in 1-indexed
        'G1896A': 1895,  # position 1896 in 1-indexed
        'G1899A': 1898,  # position 1899 in 1-indexed
        'C1653T': 1652,  # position 1653 in 1-indexed
        'T1753V': 1752,  # position 1753 in 1-indexed
        'G1613A': 1612,  # position 1613 in 1-indexed (additional)
        'C1766T': 1765,  # position 1766 in 1-indexed (additional)
        'T1768A': 1767,  # position 1768 in 1-indexed (additional)
    }
    
    # Check each position
    for name, pos in positions.items():
        if pos < len(query_list) and pos < len(ref_list):
            query_base = query_list[pos]
            ref_base = ref_list[pos]
            
            if name == 'T1753V':
                # T1753V: T mutated to any other base (A, C, or G)
                if ref_base == 'T' and query_base != 'T' and query_base != '-':
                    features[name] = 1
                else:
                    features[name] = 0
            elif name in ['PreS1', 'PreS2']:
                # These are handled separately
                pass
            else:
                # Standard point mutations
                if ref_base != query_base and query_base != '-':
                    features[name] = 1
                else:
                    features[name] = 0
    
    # Check PreS1 deletion (positions 2848-3102 in 1-indexed)
    # In 0-indexed: 2847 to 3101
    pres1_start, pres1_end = 2847, 3101
    pres1_deleted = False
    if pres1_end < len(query_list):
        pres1_region = query_list[pres1_start:pres1_end+1]
        gap_count = pres1_region.count('-')
        if gap_count > len(pres1_region) * 0.5:  # More than 50% gaps = deletion
            pres1_deleted = True
    
    # Check PreS2 deletion (positions 3103-3182 in 1-indexed)
    # In 0-indexed: 3102 to 3181
    pres2_start, pres2_end = 3102, 3181
    pres2_deleted = False
    if pres2_end < len(query_list):
        pres2_region = query_list[pres2_start:pres2_end+1]
        gap_count = pres2_region.count('-')
        if gap_count > len(pres2_region) * 0.5:
            pres2_deleted = True
    
    features['PreS1'] = 1 if pres1_deleted else 0
    features['PreS2'] = 1 if pres2_deleted else 0
    
    # Separate additional mutations (not used in model)
    additional['G1613A'] = features.pop('G1613A', 0)
    additional['C1766T'] = features.pop('C1766T', 0)
    additional['T1768A'] = features.pop('T1768A', 0)
    
    return features, additional

# Load model
def load_model():
    """Load the trained Random Forest model"""
    import pickle
    with open('hcc_new model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Main prediction function
def predict_hcc_risk(sequence):
    """Main function to predict HCC risk from HBV sequence"""
    
    # Load model
    model = load_model()
    
    # Get reference
    reference = get_reference()
    
    # Clean input
    cleaned_seq = clean_sequence(sequence)
    
    # Align
    aligned = align_sequences(cleaned_seq, reference)
    
    # Detect genotype
    genotype = detect_genotype(aligned)
    
    # Extract mutations
    features, additional = extract_features(aligned, reference)
    
    # Prepare features for model (order must match training)
    feature_order = ['A1762T', 'G1764A', 'G1896A', 'G1899A', 'C1653T', 'T1753V', 'PreS1', 'PreS2']
    feature_vector = np.array([[features[f] for f in feature_order]])
    
    # Predict
    risk_score = model.predict_proba(feature_vector)[0][1]
    risk_class = model.predict(feature_vector)[0]
    
    # Calculate confidence
    confidence = round(risk_score * 100, 1)
    
    # Prepare mutation details for display
    mutation_details = {}
    for mut in feature_order:
        status = "Present" if features[mut] == 1 else "Absent"
        mutation_details[mut] = status
    
    additional_details = {}
    for mut, value in additional.items():
        additional_details[mut] = "Present" if value == 1 else "Absent"
    
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
