# Technical Whitepaper: ML Methods for Age Classification, DARPA TA2 Challenge
Subject: Age Group Estimation via Multi-Modal Aortic and Brachial Waveform Analysis
Data Scope: 3,500 Subjects; 336-step time-series readings
Primary Metrics: Classification Accuracy and R2 (Ordinal Error Magnitude)

### 1. Executive Summary
This technical whitepaper details an investigation into machine learning methodologies for estimating age groups from physiological pressure waveforms. By leveraging a 3,500-subject dataset of aorta and brachial readings, several architectures were evaluated, including multi-branch convolutional neural networks (CNNs), hybrid ensembles, and feature-engineered gradient boosting models. The optimal performance was achieved using an XGBoost model trained on 200 raw clinical features, yielding a 64.9% accuracy and a 74.9 R2 score. This approach provides a balance of high precision and minimized ordinal error, essential for clinical aging assessments.

### 2. Exploratory Data Analysis & Clinical Feature Engineering
The foundational phase focused on extracting high-fidelity features that map physiological changes to chronological age. Waveform morphology, rather than simple pressure levels, proved to be the most significant predictor.
Key Clinical Indicators Identified:
- Arterial Stiffness (Augmentation Index): In older subjects, reflected pressure waves arrive during systole rather than diastole, augmenting the peak. The ratio of augmentation to pulse pressure was a key aging marker.
- Vascular Compliance (Dicrotic Notch): Analysis of the notch between systolic and diastolic phases revealed deeper prominence in younger subjects, correlating with higher arterial compliance.
- Spectral Energy Shifts: Frequency domain analysis indicated a shift from high to low frequencies as age increased. The spectral centroid dropped significantly from ~92 Hz in the 20s to ~77 Hz in the 70s, as waveforms "smoothed" out with age.
- Derivative Indices: The b/a ratio from the second derivative was utilized as a validated index of vascular aging.
- Cross-Sensor Dynamics: Transfer function features were calculated to evaluate the hemodynamic relationship between aortic and brachial measurement sites.

### 3. Model Architectures & Experimentation
#### 3.1 Late Fusion CNN (Temporal Feature Extraction)
To evaluate the efficacy of automated feature learning, a multi-branch "Late Fusion" CNN was developed to process aorta, brachial, difference, and ratio waves (336 x 4 matrix).
Architecture Specification:
- Multi-Scale Receptive Fields: Parallel branches with kernel sizes of 3, 7, and 11 were used to capture local and global temporal dependencies.
- Stage Extraction: A 3-stage pyramid structure (16-32-64 kernels) with Batch Normalization and ResNet-style residual connections stabilized training and built moderate-to-complex feature hierarchies.
- Feature Fusion: Global Max and Global Mean Pooling were applied to each branch, with outputs concatenated before being fed to a dense classification head (64 > 64 > 6).

#### 3.2 Hybrid Ensemble Strategies
Two hybrid architectures were designed to fuse the deep learning features with classical classification logic:
- Feature-Level Fusion: Combined XGBoost probabilities (trained on raw features) with the 64-dimensional latent features extracted from the CNN’s final hidden layer.
- Probability-Level Fusion: Combined the output probabilities from both the XGBoost and CNN models (12 total features) to learn an optimal out-of-sample balance.

#### 3.3 Iterative Feature Pruning
To enhance model generalizability, a greedy backward feature removal process was implemented. Starting with the full 200-feature set, features were iteratively removed if their absence improved cross-validated accuracy. This reduced the feature set to 173 dimensions, removing 27 noise-heavy or redundant features.

### 4. Results & Performance Evaluation
The table below summarizes the performance of all major experimental configurations evaluated on a fixed 500-subject holdout set.
| Model Architecture                         | Accuracy (%) | R2 Score | Notes                                             |
| ------------------------------------------ | -----------: | -------: | ------------------------------------------------- |
| XGBoost (Raw Features)                     |         64.9 |     74.9 | Best balanced performer; selected for production. |
| Ensemble (XGB Probs + CNN Features)        |         64.5 |     74.9 | Near-equivalent performance; higher complexity.   |
| XGBoost (Iterative Pruning - 173 Features) |         63.1 |     74.8 | Improved stability for simpler models.            |
| Ensemble (XGB Probs + CNN Probs)           |         61.8 |     75.1 | Highest R2; lowest ordinal error severity.        |
| Late Fusion CNN (Standalone)               |         53.5 |     74.4 | Learned temporal features; lower precision.       |
| Logistic Regression (Baseline)             |         51.2 |     70.2 | Baseline for linear separability.                 |


### 5. Final Model Justification & Conclusion
The XGBoost on Raw Features model was selected for the final DARPA challenge submission. While the ensembled models offered competitive R2 scores, the XGBoost model achieved the highest raw accuracy (64.9%) with significantly lower computational and architectural overhead (~275k parameters in the CNN vs. a lightweight tree-based structure).
The success of this model is largely attributed to the inclusion of domain-specific features—specifically the Augmentation Index and Spectral Centroid shifts. The results demonstrate that while deep learning (CNN) is effective at capturing general signal behavior, the integration of classical physiological markers remains paramount for high-precision age classification in medical telemetry.
