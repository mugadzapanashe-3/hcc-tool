---
title: HBV HCC Prediction Tool
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# HBV-HCC Early Detection Tool

Predicts HCC risk from HBV sequences using mutation analysis and synergy detection.

## Features

- Detects 8 key HBV mutations
- Identifies synergistic mutation pairs
- Predicts HIGH/LOW risk with confidence score
- Detects HBV genotype (A, B, C, D, F)

## Model Performance

- Accuracy: 73%
- Precision for HCC: 80%
- Trained on 140 sequences

## Synergy Pairs Detected

| Mutation Pair | Extra Risk |
|---------------|------------|
| G1764A + G1899A | +55.7% |
| PreS1 + PreS2 | +52.6% |
| G1896A + PreS1 | +49.2% |

## Use it here

Paste an HBV sequence and click Analyse.