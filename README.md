# No-Vibe-Without-Comprehension

This repository contains data, scripts, and modeling results from two experimental studies (Study 0 and Study 0+) focused on predicting code understanding using physiological signals (EEG, eye tracking, HRV) and contextual metadata.

---

## 📁 Repository Structure

```plaintext
├── Data/
│   ├── Study 0/
│   │   ├── EEG Data/           # EEG signal files (CSV format, per participant/task)
│   │   ├── Eye Data/           # Eye tracking data (CSV: fixations, timestamps, coordinates)
│   │   ├── HRV Data/           # Heart rate variability signals (CSV)
│   │   └── Volunteer Data/     # Demographics, background, comprehension scores (CSV)
│   ├── Study 0+/
│   │   ├── Code Snippets/      # Code shown to participants in Study 0+ (PNG or TXT)
│   │   ├── EEG Data/
│   │   ├── Eye Data/
│   │   ├── HRV Data/
│   │   ├── Volunteer Data/
│   │   └── Utils/              # Scripts for preprocessing and synchronization
│   └── combined_study_data.csv # Merged dataset with features and labels used for modeling
│
├── results/
│   ├── Modeling_Results_with_Confidence.csv  # Main metrics and CI estimates for models
│   └── train-test balance.png                # Distribution of classes across train/test
│
├── src/
│   ├── utils/               # Helper functions for feature extraction, metrics, CI computation
│
├── README.md                # This file
