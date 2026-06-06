from Bio import SeqIO
from Bio.Align import PairwiseAligner
import pickle
import pandas as pd
import io

def get_reference():
    with open("ref sequence.fasta", "r") as f:
        content = f.read()
    record = SeqIO.read(io.StringIO(content), "fasta")
    return str(record.seq).upper()

def clean_sequence(raw):
    lines = raw.strip().splitlines()
    cleaned = []
    for line in lines:
        if not line.startswith('>'):
            cleaned.append(line.strip())
    return ''.join(cleaned).upper()

def align_sequences(query, reference):
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    alignments = aligner.align(reference, query)
    return next(iter(alignments))

def get_aligned_sequences(alignment):
    ref_aligned = str(alignment[0])
    query_aligned = str(alignment[1])
    return ref_aligned, query_aligned

def get_base_at(query_aligned, ref_to_align, pos):
    if pos not in ref_to_align:
        return None
    align_pos = ref_to_align[pos]
    if align_pos >= len(query_aligned):
        return None
    base = query_aligned[align_pos]
    if base in ['-', 'N']:
        return None
    return base

def detect_genotype(query_aligned, ref_to_align):
    # Genotype signature positions based on Kramvis et al. and Norder et al.
    # Each genotype gets a score based on matching signature bases
    scores = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}

    # Position 1858: C=Genotype B, T=Genotype C/D/A/F
    base_1858 = get_base_at(query_aligned, ref_to_align, 1858)
    if base_1858 == 'C':
        scores['B'] += 3
    elif base_1858 == 'T':
        scores['C'] += 2
        scores['D'] += 2
        scores['A'] += 2
        scores['F'] += 2

    # Position 2507: T=Genotype B/C, A=Genotype A/D/F
    base_2507 = get_base_at(query_aligned, ref_to_align, 2507)
    if base_2507 == 'T':
        scores['B'] += 2
        scores['C'] += 2
    elif base_2507 == 'A':
        scores['A'] += 2
        scores['D'] += 2
        scores['F'] += 2

    # Position 2579: G=Genotype B/C/D, A=Genotype A/F
    base_2579 = get_base_at(query_aligned, ref_to_align, 2579)
    if base_2579 == 'G':
        scores['B'] += 1
        scores['C'] += 1
        scores['D'] += 1
    elif base_2579 == 'A':
        scores['A'] += 2
        scores['F'] += 2

    # Position 2762: G=most genotypes, distinguish by combination
    base_2762 = get_base_at(query_aligned, ref_to_align, 2762)
    if base_2762 == 'G':
        scores['A'] += 1
        scores['B'] += 1
        scores['C'] += 1
        scores['D'] += 1

    # Position 3055: A=Genotype F signature
    base_3055 = get_base_at(query_aligned, ref_to_align, 3055)
    if base_3055 == 'A':
        scores['F'] += 3
    elif base_3055 == 'G':
        scores['A'] += 1
        scores['B'] += 1
        scores['C'] += 1
        scores['D'] += 1

    # Position 621: used to distinguish A from D
    base_621 = get_base_at(query_aligned, ref_to_align, 621)
    if base_621 == 'T':
        scores['D'] += 2
    elif base_621 == 'G':
        scores['A'] += 2
        scores['C'] += 1

    # Position 513: used to distinguish genotypes
    base_513 = get_base_at(query_aligned, ref_to_align, 513)
    if base_513 == 'A':
        scores['A'] += 2
        scores['F'] += 1
    elif base_513 == 'C':
        scores['B'] += 1
        scores['C'] += 1
        scores['D'] += 1

    # Find best scoring genotype
    best_genotype = max(scores, key=scores.get)
    best_score = scores[best_genotype]

    if best_score == 0:
        return 'Unknown'

    # Check if it is a clear winner or ambiguous
    sorted_scores = sorted(scores.values(), reverse=True)
    if sorted_scores[0] == sorted_scores[1]:
        return 'Ambiguous'

    return best_genotype

def extract_features(query_sequence):
    print("Fetching reference sequence...")
    reference = get_reference()

    print("Cleaning input sequence...")
    clean_query = clean_sequence(query_sequence)

    if len(clean_query) < 100:
        raise Exception("Sequence is too short. Please paste a complete HBV genome sequence.")

    print("Aligning sequences... please wait")
    alignment = align_sequences(clean_query, reference)
    ref_aligned, query_aligned = get_aligned_sequences(alignment)

    ref_pos = 0
    ref_to_align = {}
    for i, base in enumerate(ref_aligned):
        if base != '-':
            ref_pos += 1
            ref_to_align[ref_pos] = i

    # Detect genotype
    genotype = detect_genotype(query_aligned, ref_to_align)

    # --- Model features (used for prediction) ---
    model_features = {}

    base = get_base_at(query_aligned, ref_to_align, 1762)
    model_features['A1762T'] = 1 if base == 'T' else 0

    base = get_base_at(query_aligned, ref_to_align, 1764)
    model_features['G1764A'] = 1 if base == 'A' else 0

    base = get_base_at(query_aligned, ref_to_align, 1896)
    model_features['G1896A'] = 1 if base == 'A' else 0

    base = get_base_at(query_aligned, ref_to_align, 1899)
    model_features['G1899A'] = 1 if base == 'A' else 0

    base = get_base_at(query_aligned, ref_to_align, 1653)
    model_features['C1653T'] = 1 if base == 'T' else 0

    base = get_base_at(query_aligned, ref_to_align, 1753)
    model_features['T1753V'] = 1 if (base is not None and base in ['G', 'C', 'A']) else 0

    pres1_start = ref_to_align.get(2848)
    pres1_end = ref_to_align.get(3102)
    if pres1_start is not None and pres1_end is not None:
        pres1_region = query_aligned[pres1_start:pres1_end+1]
        model_features['PreS1'] = 1 if '-' in pres1_region else 0
    else:
        model_features['PreS1'] = 0

    pres2_start = ref_to_align.get(3103)
    pres2_end = ref_to_align.get(3182)
    if pres2_start is not None and pres2_end is not None:
        pres2_region = query_aligned[pres2_start:pres2_end+1]
        model_features['PreS2'] = 1 if '-' in pres2_region else 0
    else:
        model_features['PreS2'] = 0

    # --- Additional informational features ---
    additional_features = {}

    base = get_base_at(query_aligned, ref_to_align, 1613)
    additional_features['G1613A'] = 1 if base == 'A' else 0

    base = get_base_at(query_aligned, ref_to_align, 1766)
    additional_features['C1766T'] = 1 if base == 'T' else 0

    base = get_base_at(query_aligned, ref_to_align, 1768)
    additional_features['T1768A'] = 1 if base == 'A' else 0

    return model_features, additional_features, genotype

def predict_hcc_risk(sequence):
    model_features, additional_features, genotype = extract_features(sequence)
    print("Model mutations detected:", model_features)
    print("Additional mutations detected:", additional_features)
    print("Genotype detected:", genotype)

    model = pickle.load(open('hcc_new model.pkl', 'rb'))

    columns = ['A1762T','G1764A','G1896A','G1899A','C1653T','T1753V','PreS1','PreS2']
    input_data = pd.DataFrame([model_features], columns=columns)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        result = "HIGH RISK — HCC associated mutations detected"
    else:
        result = "LOW RISK — No significant HCC associated mutations detected"

    confidence = round(max(probability) * 100, 2)
    return result, confidence, model_features, additional_features, genotype
   
