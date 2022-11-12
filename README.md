# scaling-uod
Repository for the paper "Scaling Unsupervised Outlier Detection: industrial challenges and an effective approach".

## Data sets
Data used in the paper is contained in the **data/** folder.

The data sets in this repo are represented using the [ARFF format](https://www.cs.waikato.ac.nz/~ml/weka/arff.html). The first attribute 'id' is a unique row identifier. The second attribute 'outlier' is the target. A value of 0.0 means normal. A value of 1.0 means outlier.

## Installing and running

Install the software with:

```bash
poetry install
```

Run the software with:
```bash
poetry run scaling-uod-poc
```
