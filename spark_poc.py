import logging
import numpy as np
import os
import pyspark.sql.functions as sf
import pyspark.sql.types as st
import shutil

from pyspark.sql import Row, SparkSession
from sklearn.neighbors import LocalOutlierFactor


logging.basicConfig(
    level='INFO',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def add_partition_number(df, partition_size, seed):
    '''Assigns a random partition number to each data elements in a data set.

    The data set is represented as a Spark DataFrame.

    Partition numbers are assigned so that each partition size is approximately 
    equal to partition_size.
    '''
    data_size = df.count()
    num_components = max(1, data_size // partition_size)
    df = df.withColumn(
        'component_id',
        (sf.rand(seed) * num_components).cast(st.IntegerType())
    )
    return df


def get_partition_function(k):
    '''Defines the function to be applied to each partition.

    In this case the function computes the LOF score for each data element 
    in the partition.

    The parameter k is used to determine the nearest neighbors for LOF.
    '''

    def partition_function(row_itr):
        row_list = list(row_itr)
        data_matrix = row_list_to_data_matrix(row_list)
        lof = LocalOutlierFactor(
            n_neighbors=k,
            algorithm='kd_tree',
            contamination='auto',
        )       
        lof_model = lof.fit(data_matrix)
        scores = [score.item() for score in -lof_model.negative_outlier_factor_]
        rows_with_scores = [
            Row(
                **{
                    **row.asDict(),
                    **{'outlier_score': score}
                }
            )
            for row, score in zip(row_list, scores)
        ]
        return iter(rows_with_scores)

    return partition_function


def get_spark_session():
    '''Initializes the Spark Session.'''

    master = 'local[*]'
    return (
        SparkSession.builder
        .master(master)
        .appName('Proof-of-concept Spark')
        .getOrCreate()
    )


def load_data(spark, filepath):
    df = spark.read.csv(filepath, inferSchema=True)

    # Drop the first two features, containing id and outlier label, respectively.
    id_label_column_names = df.columns[:2]
    df = df.drop(*id_label_column_names) 

    return df


def row_list_to_data_matrix(row_list):
    '''Converts a list of Spark Rows to a 2-dimensional numpy.ndarray.'''   
    return np.array(
        [
            list(row.asDict().values())
            for row in row_list
        ]
    )


def repartition_unique(df):
    '''Repartitions the input DataFrame into unique partitions. Returns the repartitioned RDD.

    After the repartitioning, all data elements in a partition have the same component_id. This can be 
    extended to include any contextual features.
    '''
    
    # Select the distinct values of 'component_id' as a list of Rows
    component_ids = df.select('component_id').distinct().collect()
    num_partitions = len(component_ids)

    # Assigns to each distinct component id a unique partition number corresponding to its rank in the enumeration.
    hash_map = {}
    for i, row in enumerate(component_ids):
        hash_map[row] = i

    # Define the unique partitioner function
    def unique_partitioner(row):
        return hash_map[
            Row(component_id=row['component_id'])
        ]

    # Custom partitioning requires (key, value) RDDs
    key_value_rdd = df.rdd.map(lambda row: (row, row))  

    repartitioned_rdd = key_value_rdd.partitionBy(num_partitions, unique_partitioner)

    # Restore initial RDD, removes (key, value) representation
    repartitioned_rdd = repartitioned_rdd.map(lambda key_value_pair: key_value_pair[1])

    return repartitioned_rdd

    
def main():
    '''Proof-of-Concept of the approach described in "Scaling Unsupervised Outlier Detection: industrial challenges and an effective approach".

    Implemented using Apache Spark APIs.

    Approach:
    - load a dummy data set from an input file.
    - partition the data set with the partitioning scheme described in the paper.
    - computes the Local Outlier Factor (LOF) for each partition independently.
    - save the results to an output file.
    '''
    
    INPUT_FILE = 'data/dummy.csv'
    OUTPUT_FILE = 'results.dir/'
    
    ## PARAMETERS
    K = 10                # K parameter for the LOF method
    SEED = 42             # Random number generator seed
    PARTITION_SIZE = 100  # Desired partition size

    logging.info('Initializing Apache Spark session...')    
    spark = get_spark_session()

    logging.info('Loading data set from %s', INPUT_FILE)
    data_df = load_data(spark, INPUT_FILE)

    logging.info('Assigning data elements to partitions...')    
    data_with_partitions_df = add_partition_number(data_df, PARTITION_SIZE, SEED)

    logging.info('Partitioning RDD...')    
    repartitioned_rdd = repartition_unique(data_with_partitions_df)

    logging.info('Computing outlier scores...')    
    partition_function = get_partition_function(K)
    scores_rdd = repartitioned_rdd.mapPartitions(partition_function, preservesPartitioning=True)

    logging.info('Saving results to %s', OUTPUT_FILE)
    output_dir = os.path.dirname(OUTPUT_FILE)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    scores_df = spark.createDataFrame(scores_rdd)
    scores_df.write.csv(OUTPUT_FILE)

if __name__ == '__main__':
    main()
