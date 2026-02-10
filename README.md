# Fairness Evaluation of Audio Language Models in Speech Emotion Recognition

This repository contains the code and resources for evaluating fairness of Audio-based Large Language Models (ALMs) in Speech Emotion Recognition (SER). This work addresses a critical gap in the literature by examining whether ALMs behave equitably toward individuals across sensitive groups such as gender, age, and ethnicity.

## Overview

Recognizing emotions from speech using advanced Audio-Language Models (ALMs) has shown promise due to their zero-shot capabilities, with applications in healthcare, education, and human-computer interaction. However, fairness evaluation of ALMs across sensitive groups remains largely unexplored. 

This work evaluates **Qwen2Audio**, a competitive ALM for Speech Emotion Recognition, using fairness metrics including:
- **Statistical Parity**
- **Equal Opportunity**
- **Overall Accuracy Equality (OAE)**

Our evaluation accounts for the multiclass nature of emotions and extends fairness assessment to multigroup-multiclass scenarios, addressing two key challenges:
1. Sensitive attributes may have more than two categories (multigroup setting)
2. SER is inherently a multiclass classification task

## Key Findings

- Qwen2Audio achieves the best performance among evaluated open-source ALMs for SER
- The model exhibits **unfairness across datasets and sensitive attributes** in terms of Overall Accuracy Equality (OAE)
- Results emphasize the need for fairness-aware SER models

## Evaluated Datasets

The fairness evaluation is conducted on multiple benchmark datasets:
- **CREMA-D**
- **IEMOCAP**
- **EmoV-DB**
- **RAVDESS**
- **MELD**

## Main Contributions

1. **Performance and Fairness Analysis**: We analyze both the performance and fairness of the Qwen2-Audio ALM in SER, examining the impact of generative temperature
2. **Multiclass Fairness Assessment**: We adopt a recent approach for multiclass fairness assessment and apply it to the SER task
3. **Unfairness Characterization**: We demonstrate that unfairness in Qwen2-Audio for SER manifests in terms of Overall Accuracy Equality (OAE)

## EmoBox Benchmark

This project utilizes the **EmoBox** toolkit and benchmark for speech emotion recognition evaluation. EmoBox is an out-of-the-box multilingual multi-corpus speech emotion recognition toolkit that provides standardized data processing, evaluation metrics, and benchmarks for both intra-corpus and cross-corpus settings.

### About EmoBox

- **GitHub Repository**: [https://github.com/emo-box/EmoBox](https://github.com/emo-box/EmoBox)
- **Benchmark Website**: [https://emo-box.github.io/index.html](https://emo-box.github.io/index.html)
- **Paper**: [EmoBox: Multilingual Multi-corpus Speech Emotion Recognition Toolkit and Benchmark](https://arxiv.org/abs/2406.07162)

EmoBox includes **32 speech emotion datasets** spanning **14 distinct languages** with standardized data preparation and partitioning. For detailed information about datasets, preparation scripts, and usage examples, please refer to the [EmoBox documentation](EmoBox/README.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
