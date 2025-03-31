# cross_domain_hedge_detection

By Melisa Özdemir, Josefa De La Luz Costa Rojo, Kseniya K 

A transformer-based system for hedge cue detection across domains, supporting training, inference, and cross-domain analysis. This repository includes code for training on labeled datasets (BioScope, HedgePeer), inference on political debates (UNSC), and evaluating model performance.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Notebooks Overview for Cue Detection and Analysis

### 1. `unsc_cue_inference_data_analysis.ipynb`
**UNSC Hedge Cue Inference & Data Analysis**

This notebook performs hedge cue inference on the UN Security Council (UNSC) corpus, using a fine-tuned transformer model. It also analyzes the frequency and distribution of hedging across the corpus.

**Overview**
- **Goal**: Identify hedge cues in political debate transcripts using inference.
- **Corpus**: UN Security Council (UNSC) meeting protocols (1992–2023)
- **Model**: Fine-tuned transformer (BERT) for token classification
- **Output**: Sentence-level predictions, hedging statistics, and visualizations

**Usage**
- Run all cells in Jupyter Notebook, Kaggle, or Colab.
- To switch the model:
  - Update `model_path` and `tokenizer_path`
  - Update dataset path accordingly

---

### 2. `hedge_cue_detection_training.ipynb`
**Hedge Cue Detection – Model Training Pipeline**

This notebook implements the training pipeline for hedge cue detection using transformer-based models. Cue detection is framed as a binary token classification task.

**Overview**
- **Goal**: Train a transformer model to classify HEDGE vs NON-HEDGE tokens
- **Supported Models**: `BERT`, `SciBERT`, `XLNet`  
- **Datasets**: `HedgePeer`, `BioScope`

**Usage**
- Set the `trans_model` variable to select the transformer model
- Upload and specify the correct paths for datasets
- Training artifacts are saved to disk (.pt model + tokenizer)

---

### 3. `cue_inference_on_bioscope_hedgepeer.ipynb`
**Hedge Cue Inference – Evaluation on BioScope & HedgePeer**

Performs inference on labeled test sets from BioScope and HedgePeer using previously fine-tuned transformer models.

**Overview**
- **Goal**: Evaluate model performance on test data
- **Datasets**: `BioScope` and `HedgePeer` (labeled)
- **Model**: Fine-tuned transformer (BERT, SciBERT, XLNet)
- **Metrics**: Precision, Recall, F1-score (saved to CSV)

**Usage**
- Choose the model to evaluate (comment/uncomment as needed)
- Upload the dataset and fine-tuned `.pt` model weights
- Update paths to model and tokenizer

---

## Folder Structure 

```
cross_domain_hedge_detection/
│
├── notebooks/
│   ├── cue_notebooks
│   ├── span_notebooks
│   
│
├── requirements.txt
├── README.md

└── datasets/   
```


## Acknowledgments

This project was supported by guidance from Prof. Manfred Stede and also GPU resources were provided.

---
