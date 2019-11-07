# scaling-uod
Repository for the paper "Scaling Unsupervised Outlier Detection: industrial challenges and an effective approach".

## Data sets
Data used in the paper is contained in the **data/** folder.

The data sets in this repo are represented using the [ARFF format](https://www.cs.waikato.ac.nz/~ml/weka/arff.html). The first attribute 'id' is a unique row identifier. The second attribute 'outlier' is the target. A value of 0.0 means normal. A value of 1.0 means outlier.

## Installing and running

### Prerequisites
To run the proof-of-concept in this repo you must have python3 and pip installed.

To download the required packages run (we suggest activating a **virtualenv** before running the command):

> pip install -r requirements.txt

Also be sure to set the variable SPARK_HOME to point to a valid Apache Spark installation.

This code was tested with Apache Spark version 2.4.4. All packages versions can be found in **requirements.txt**.

### Running
After all prerequisites are installed, run the proof-of-concept with:

> python3 spark_poc.py