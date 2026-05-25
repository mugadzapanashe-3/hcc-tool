from Bio import SeqIO
import subprocess
import tempfile
import os
import pickle
import pandas as pd
import requests
import io

MAFFT_PATH = r"D:\mafft\mafft-win\mafft.bat"
REFERENCE_ID = "AB033559.1"

def get_reference():
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nucleotide&id={REFERENCE_ID}&rettype=fasta&retmode=text"
    response = requests.get(url)
    record = SeqIO.read(io.StringIO(response.text), "fasta")
    return str(record.seq).upper()

def clean_sequence(raw):
    lines = raw.strip().splitlines()
    cleaned = []
    for line in lines:
        if not line.startswith('>'):
            cleaned.append(line.strip())
    return ''.join(cleaned).upper()

def align_with_mafft(query_sequence, reference_sequence):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(f">reference\n{reference_sequence}\n")
        f.write(f">query\n{query_sequence}\n")
        input_file = f.name

    try:
        result = subprocess.run(
            [MAFFT_PATH, '--quiet', '--auto', input_file],
            capture_output=True,
            text=True,
            timeout=120
        )
        aligned_text = result.stdout

        if not aligned_text.strip():
            raise Exception("MAFFT returned empty output. Please check your sequence.")

        sequences = {}
        current_name = None
        current_seq = []
        for line in aligned_text.splitlines():
            if line.startswith('>'):
                if current_name:
                    sequences[current_name] = ''.join(current_seq)
                current_name = line[1:].strip()
                current_seq = []
            else:
                current_seq.append(line.strip())
        if current_name:
            sequences[current_name] = ''.join(current_seq)

        return sequences

    finally:
        if os.path.exists(input_file):
            os.remove(input_file)

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

def extract_features(query_sequence):
    print("Fetching reference sequence...")
    reference = get_reference()

    print("Cleaning input sequence...")
    clean_query = clean_sequence(query_sequence)

    if len(clean_query) < 100:
        raise Exception("Sequence is too short. Please paste a complete HBV genome sequence.")

    print("Aligning with MAFFT... please wait")
    sequences = align_with_mafft(clean_query, reference)

    ref_aligned = None
    query_aligned = None
    for name, seq in sequences.items():
        if 'reference' in name.lower():
            ref_aligned = seq.upper()
        else:
            query_aligned = seq.upper()

    if ref_aligned is None or query_aligned is None:
        raise Exception("Alignment failed. Could not identify reference and query sequences.")

    ref_pos = 0
    ref_to_align = {}
    for i, base in enumerate(ref_aligned):
        if base != '-':
            ref_pos += 1
            ref_to_align[ref_pos] = i

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

    # --- Additional informational features (not used in model) ---
    additional_features = {}

    base = get_base_at(query_aligned, ref_to_align, 1613)
    additional_features['G1613A'] = 1 if base == 'A' else 0

    base = get_base_at(query_aligned, ref_to_align, 1766)
    additional_features['C1766T'] = 1 if base == 'T' else 0

    base = get_base_at(query_aligned, ref_to_align, 1768)
    additional_features['T1768A'] = 1 if base == 'A' else 0

    return model_features, additional_features

def predict_hcc_risk(sequence):
    model_features, additional_features = extract_features(sequence)
    print("Model mutations detected:", model_features)
    print("Additional mutations detected:", additional_features)

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
    return result, confidence, model_features, additional_features