# Lie Equivariant Perception Algebraic Unified Transform Embedding Framework (L.E.P.A.U.T.E. Framework)

A Python package for processing webcam images with a Lie group-based Transformer model and accessing the resulting data.

## Installation

```
pip install lepaute
```


## Usage

Run the [example_pip.py](example_pip.py) file.

## Requirements

- Python >= 3.8
- torch >= 1.12.0
- kornia == 0.7.0
- opencv-python
- numpy

## Notes

- Ensure webcam access for real-time data collection.
- In Pyodide, data is stored in memory only.
- Debug logs can be enabled by setting `logging.basicConfig(level=logging.DEBUG)`.
