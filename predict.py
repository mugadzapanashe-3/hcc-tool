import pandas as pd
import numpy as np
from Bio import Align
import pickle

# ─────────────────────────────────────────────
# SYNERGY DEFINITIONS
# ─────────────────────────────────────────────

def get_synergies():
    """
    Synergistic mutation pairs identified via logistic regression analysis.
    Synergy is defined as an increase in HCC rate >10% above additive expectation.
    """
    return [
        {"pair": "G1764A + G1899A", "synergy_effect": 55.7},
        {"pair": "PreS1 + PreS2",   "synergy_effect": 52.6},
        {"pair": "G1896A + PreS1",  "synergy_effect": 49.2},
        {"pair": "G1899A + C1653T", "synergy_effect": 45.8},
        {"pair": "A1762T + T1753V", "synergy_effect": 32.9},
    ]

# Threshold above which synergy alone upgrades risk to HIGH
SYNERGY_UPGRADE_THRESHOLD = 40.0

# ─────────────────────────────────────────────
# REFERENCE & SEQUENCE UTILITIES
# ─────────────────────────────────────────────

def get_reference():
    """Load reference sequence from FASTA file."""
    with open('ref sequence.fasta', 'r') as f:
        lines = f.readlines()
    sequence = ''.join([line.strip() for line in lines if not line.startswith('>')])
    return sequence.upper()


def clean_sequence(raw_sequence):
    """Strip FASTA headers and whitespace from input sequence."""
    lines = raw_sequence.strip().split('\n')
    cleaned = ''.join([line.strip() for line in lines if not line.startswith('>')])
    cleaned = ''.join(cleaned.split())
    return cleaned.upper()


# ─────────────────────────────────────────────
# ALIGNMENT
# ─────────────────────────────────────────────

def align_sequences(query, reference):
    """Align query against reference using BioPython PairwiseAligner (global)."""
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -2

    alignments = aligner.align(reference, query)
    best = alignments[0]
    return best[1]


# ─────────────────────────────────────────────
# GENOTYPE DETECTION
# ─────────────────────────────────────────────

def detect_genotype(sequence):
    """Heuristic genotype detection based on positional nucleotide signatures."""
    genotype_scores = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}

    # Genotype A
    if len(sequence) > 97  and sequence[0]   == 'T': genotype_scores['A'] += 1
    if len(sequence) > 53  and sequence[52]  == 'C': genotype_scores['A'] += 1
    if len(sequence) > 97  and sequence[96]  == 'C': genotype_scores['A'] += 1

    # Genotype B
    if len(sequence) > 52  and sequence[51]  == 'C': genotype_scores['B'] += 1
    if len(sequence) > 96  and sequence[95]  == 'A': genotype_scores['B'] += 1
    if len(sequence) > 127 and sequence[126] == 'A': genotype_scores['B'] += 1
    if len(sequence) > 25  and sequence[24]  == 'T': genotype_scores['B'] += 1
    if len(sequence) > 147 and sequence[146] == 'T': genotype_scores['B'] += 1

    # Genotype C
    if len(sequence) > 76  and sequence[75]  == 'C': genotype_scores['C'] += 1
    if len(sequence) > 165 and sequence[164] == 'C': genotype_scores['C'] += 1
    if len(sequence) > 10  and sequence[9]   == 'A': genotype_scores['C'] += 1
    if len(sequence) > 49  and sequence[48]  == 'A': genotype_scores['C'] += 1

    # Genotype D
    if len(sequence) > 43  and sequence[42]  == 'A': genotype_scores['D'] += 1
    if len(sequence) > 55  and sequence[54]  == 'C': genotype_scores['D'] += 1
    if len(sequence) > 135 and sequence[134] == 'T': genotype_scores['D'] += 1
    if len(sequence) > 148 and sequence[147] == 'G': genotype_scores['D'] += 1

    # Genotype F
    if len(sequence) > 9   and sequence[8]  == 'A': genotype_scores['F'] += 1
    if len(sequence) > 4   and sequence[3]  == 'A': genotype_scores['F'] += 1
    if len(sequence) > 8   and sequence[7]  == 'C': genotype_scores['F'] += 1
    if len(sequence) > 19  and sequence[18] == 'G': genotype_scores['F'] += 1

    best = max(genotype_scores, key=genotype_scores.get)
    return best if genotype_scores[best] > 0 else 'Unknown'


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extract_features(aligned_query, reference):
    """
    Extract presence/absence of HCC-associated HBV mutations
    from the aligned query sequence.
    """
    features    = {}
    additional  = {}

    q = list(aligned_query)
    r = list(reference)

    def point_mut(pos, ref_base, alt_base):
        """Return 1 if mutation is present at 0-indexed position."""
        if pos < len(q):
            return 1 if q[pos] == alt_base and r[pos] == ref_base else 0
        return 0

    def deletion_mut(start, end):
        """Return 1 if >50% of region is gap-deleted."""
        if end < len(q):
            region = q[start:end + 1]
            return 1 if region.count('-') > len(region) * 0.5 else 0
        return 0

    # Core model features
    features['A1762T'] = point_mut(1761, 'A', 'T')
    features['G1764A'] = point_mut(1763, 'G', 'A')
    features['G1896A'] = point_mut(1895, 'G', 'A')
    features['G1899A'] = point_mut(1898, 'G', 'A')
    features['C1653T'] = point_mut(1652, 'C', 'T')
    features['T1753V'] = 1 if (1752 < len(q) and r[1752] == 'T'
                                and q[1752] != 'T' and q[1752] != '-') else 0
    features['PreS1']  = deletion_mut(2847, 3101)
    features['PreS2']  = deletion_mut(3102, 3181)

    # Additional mutations (informational, not fed into Random Forest)
    additional['G1613A'] = point_mut(1612, 'G', 'A')
    additional['C1766T'] = point_mut(1765, 'C', 'T')
    additional['T1768A'] = point_mut(1767, 'T', 'A')

    return features, additional


# ─────────────────────────────────────────────
# SYNERGY ANALYSIS
# ─────────────────────────────────────────────

def analyse_synergies(features):
    """
    Check which synergistic mutation pairs are both present.

    Returns:
        detected_synergies  : list of dicts with pair name, extra_risk, and clinical note
        total_synergy_boost : sum of all active synergy effects (capped at 99%)
        synergy_upgrade     : bool — True if any single pair exceeds SYNERGY_UPGRADE_THRESHOLD
    """
    detected_synergies = []
    total_synergy_boost = 0.0
    synergy_upgrade = False

    for pair in get_synergies():
        mut1, mut2 = pair['pair'].split(' + ')
        if features.get(mut1, 0) == 1 and features.get(mut2, 0) == 1:

            effect = pair['synergy_effect']
            total_synergy_boost += effect

            # Assign clinical urgency level
            if effect >= 50:
                urgency = 'CRITICAL'
                note = (f"{mut1} and {mut2} together produce a critically elevated "
                        f"synergistic HCC risk (+{effect}%). Immediate clinical "
                        f"surveillance is strongly recommended.")
            elif effect >= 40:
                urgency = 'HIGH'
                note = (f"{mut1} and {mut2} co-occurrence elevates HCC risk by "
                        f"+{effect}% above additive expectation. Close monitoring advised.")
            else:
                urgency = 'MODERATE'
                note = (f"{mut1} and {mut2} show moderate synergistic interaction "
                        f"(+{effect}%). Consider increased surveillance frequency.")

            if effect >= SYNERGY_UPGRADE_THRESHOLD:
                synergy_upgrade = True

            detected_synergies.append({
                'pair':       pair['pair'],
                'extra_risk': effect,
                'urgency':    urgency,
                'note':       note,
            })

    # Sort by highest synergy effect first
    detected_synergies.sort(key=lambda x: x['extra_risk'], reverse=True)

    # Cap total boost at 99 to avoid exceeding 100% when added to base score
    total_synergy_boost = min(total_synergy_boost, 99.0)

    return detected_synergies, total_synergy_boost, synergy_upgrade


# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────

def load_model():
    """Load the trained Random Forest model."""
    with open('hcc_new model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


# ─────────────────────────────────────────────
# MAIN PREDICTION FUNCTION
# ─────────────────────────────────────────────

def predict_hcc_risk(sequence):
    """
    Full HCC risk prediction pipeline:
      1. Clean & align input sequence
      2. Detect genotype
      3. Extract mutation features
      4. Random Forest base prediction
      5. Synergy analysis & risk upgrade
      6. Return structured result dict
    """

    model      = load_model()
    reference  = get_reference()
    cleaned    = clean_sequence(sequence)
    aligned    = align_sequences(cleaned, reference)
    genotype   = detect_genotype(aligned)
    features, additional = extract_features(aligned, reference)

    # ── Base Random Forest prediction ──────────────────────────────────────
    feature_order  = ['A1762T', 'G1764A', 'G1896A', 'G1899A',
                      'C1653T', 'T1753V', 'PreS1', 'PreS2']
    feature_vector = np.array([[features[f] for f in feature_order]])

    base_risk_score  = model.predict_proba(feature_vector)[0][1]   # probability 0–1
    base_risk_class  = model.predict(feature_vector)[0]            # 0 or 1
    base_confidence  = round(base_risk_score * 100, 1)

    # ── Synergy analysis ───────────────────────────────────────────────────
    detected_synergies, total_synergy_boost, synergy_upgrade = analyse_synergies(features)

    # ── Adjusted confidence score ──────────────────────────────────────────
    # Add synergy boost proportionally (synergy contribution scaled to 30% weight)
    synergy_contribution  = total_synergy_boost * 0.30
    adjusted_confidence   = min(round(base_confidence + synergy_contribution, 1), 99.9)

    # ── Final risk label ───────────────────────────────────────────────────
    # Upgrade to HIGH RISK if:
    #   (a) Random Forest already says HIGH, OR
    #   (b) A strong synergistic pair (>= threshold) is detected
    if base_risk_class == 1 or synergy_upgrade:
        final_risk  = 'HIGH RISK'
        risk_label  = 1
    else:
        final_risk  = 'LOW RISK'
        risk_label  = 0

    # ── Synergy override warning message ──────────────────────────────────
    synergy_warning = None
    if synergy_upgrade and base_risk_class == 0:
        top_pair = detected_synergies[0]
        synergy_warning = (
            f"⚠️ SYNERGY OVERRIDE: Individual mutations suggested LOW RISK, "
            f"however synergistic interaction between {top_pair['pair']} "
            f"(+{top_pair['extra_risk']}% above additive expectation) has "
            f"upgraded this result to HIGH RISK. Clinical monitoring is recommended."
        )

    # ── Mutation display dicts ─────────────────────────────────────────────
    mutation_details    = {m: ("Present" if features[m] == 1 else "Absent")
                           for m in feature_order}
    additional_details  = {m: ("Present" if additional[m] == 1 else "Absent")
                           for m in additional}

    return {
        'risk':                 final_risk,
        'base_confidence':      base_confidence,
        'adjusted_confidence':  adjusted_confidence,
        'synergy_boost':        round(total_synergy_boost, 1),
        'synergy_warning':      synergy_warning,
        'genotype':             genotype,
        'mutations':            mutation_details,
        'additional_mutations': additional_details,
        'synergies':            detected_synergies,
    }
 
 
