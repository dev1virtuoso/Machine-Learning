# Usage Guide

## 1. Overview
The Linguistic Analysis and Knowledge Extraction System (L.A.K.E.S.) operates in two distinct operational modes: training and inference. The system processes raw, unstructured text through a deterministic multi-stage linguistic parser and maps the resulting structural components into neural sequence estimators to identify entities, relational boundaries, and event-role mappings.

## 2. Install Requirements
Prior to executing the pipeline modules, the required runtime environment and language resources must be provisioned.

Execute the following package management commands to prepare the environment:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Manual Environment and Parameter Configuration

Before executing the pipeline, several structural variables, path parameters, and environment dependencies must be configured manually inside the source files or specified via runtime arguments:

1. **Stanford CoreNLP Directory Path (`--corenlp_path`)**
* **Parameter:** `corenlp_path` in `main.py` (Default: `/opt/stanford-corenlp-full-2018-02-27`)
* **Action Required:** Download the Stanford CoreNLP package (version 3.8.0) and its corresponding English models jar. Update this path parameter to point to the absolute directory where the files were unpacked.


2. **Java Runtime Environment Memory Allocation**
* **Parameter:** `memory` parameter inside the `DeepLinguisticAnalyzer` initialization within `linguistic_pipeline.py` (Default: `"4g"`)
* **Action Required:** Ensure that a Java Runtime Environment (JRE) is accessible via the system environment variables (`PATH`). If processing documents exceeding 40,000 characters, manually increase this allocation string (e.g., `"8g"` or `"16g"`) to prevent out-of-memory crashes within the Stanford CoreNLP local server instance.


3. **Label Map Dictionaries (`tag_to_ix`, `num_role_tags`)**
* **Parameter:** `tag_to_ix` mappings passed to `BiLSTM_CRF` and `JointEventExtractor`
* **Action Required:** If modifying the training dataset schemas, you must explicitly populate the `Vocabulary` object or manually inject special tokens (`<START>`, `<STOP>`, `<PAD>`, `O`) into the label indexers to match the exact dimensions of your custom categorical target sets.


4. **Network Dimension Hyperparameters**
* **Parameters:** `--embedding_dim` (Default: `100`), `--hidden_dim` (Default: `256`)
* **Action Required:** Ensure that pre-trained embedding layers or downstream linear projections match your intended vector shapes. If utilizing specific pre-trained weights, the `--embedding_dim` must be manually overridden to match the exact channel dimensions of the source matrix.

### Operational Execution

#### Training Pipeline Execution

To execute model training over a structured, weakly supervised, or manually annotated dataset, launch `main.py` in training mode:

```bash
python main.py --mode train --train_file path/to/dataset.jsonl --save_model model.pth --embedding_dim 100 --hidden_dim 256 --batch_size 32 --epochs 10
```

#### Inference Pipeline Execution

To execute end-to-end extraction over a raw text string using a saved model checkpoint, launch the orchestrated pipeline in inference mode:

```bash
python main.py --mode infer --text "The deep network framework extracted semantic primitives smoothly." --load_model model.pth --corenlp_path /path/to/stanford-corenlp
```

## 3. Troubleshooting

### Runtime CUDA Asset Allocation Faults

* **Symptom:** `TypeError` or `RuntimeError` regarding tensor type mismatches when moving sequence lengths or data structures across GPU and host devices.
* **Cause:** The 2017-era PyTorch 0.3.1 framework handles tensor indices rigidly via `Variable` abstractions. In certain modules, explicit `.cuda()` calls are bypassed if the input variables are constructed dynamically mid-forward pass.
* **Resolution:** Ensure the flag `--no_cuda` is passed if your local environment lacks a matching CUDA 8.0/9.0 runtime toolkit, or explicitly wrap runtime-generated index tensors (such as `sort_idx` or `unsort_idx`) in `.cuda()` bounds inside `bilstm_crf.py` and `relation_extractor.py`.

### CoreNLP Server Connection Expiration or Chunk Dropouts

* **Symptom:** `StanfordCoreNLP` drops connection sockets during high-throughput document streaming, resulting in partial fallback states or persistent exception loops.
* **Cause:** Large documents block the single-threaded local socket allocation. While `DeepLinguisticAnalyzer` breaks inputs down into a maximum character ceiling of 40,000 elements, overlapping sentence windows can cause high processing latency.
* **Resolution:** Increase the initialization timeout parameter in `linguistic_pipeline.py` past `60000` milliseconds, or reduce the maximum character ceiling limit within the `_get_sentence_chunks` method block.

### PyTorch Index Selection and Sequence Padding Errors

* **Symptom:** Errors arising from `pack_padded_sequence` regarding unsorted or zero-length sequence entries.
* **Cause:** The neural sequence wrapper requires batches to be sorted by sequence length in strict descending order. While sorting transforms are applied via `sort_idx` inside `_get_lstm_features`, if an empty token slice occurs due to sentence-splitting edge cases, the utility will fail.
* **Resolution:** Filter your input text data to remove empty strings or lines containing only whitespaces prior to dispatching them to the linguistic extraction pipeline.