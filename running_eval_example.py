#!/usr/bin/env python3
import timeeval
from pathlib import Path
import pandas as pd
import numpy as np
from timeeval.utils.hash_dict import hash_dict
import json
from timeeval import TimeEval, MultiDatasetManager, Algorithm, TrainingType, InputDimensionality, ResourceConstraints
from timeeval.adapters import DockerAdapter
from timeeval import DefaultMetrics
from timeeval.metrics import RangePrAUC,RangeRocVUS, RangePrVUS
from timeeval.resource_constraints import GB
from timeeval.utils.window import ReverseWindowing
from timeeval import TimeEval, RemoteConfiguration, ResourceConstraints
from durations import Duration

# post-processing for TDP_AD, from windows score to observations score
def post_tdp_ad(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("s", 20)
    return ReverseWindowing(window_size=window_size).fit_transform(scores)


def main(cpu):

    # cluster setup
    cluster_config = RemoteConfiguration(
    scheduler_host="master", # master adresse
    worker_hosts=[ 
        "worker1", # add workers adresses 
        ]
    )
    
    # the path of the dataset  collections   file (dataset.csv) 
    dm = MultiDatasetManager([Path("/home/ubuntu/timeeval-datasets/")]) 

    datasets = dm.select() 
    

    algorithms = [
         Algorithm(
           name= "TDP",
           main=DockerAdapter(image_name="registry.gitlab.hpi.de/akita/i/tdp",tag="1", skip_pull=True),
           data_as_file=True,
           postprocess= post_tdp_ad,
           training_type=TrainingType.UNSUPERVISED,
           input_dimensionality=InputDimensionality.MULTIVARIATE
        )
         ]
    

    repetitions = 15
    rcs = ResourceConstraints(
    task_memory_limit = 3 * GB,
    task_cpu_limit = cpu, 
    tasks_per_host=1,
    execute_timeout=Duration(" 10 hours")
)
     
    metrics = [
    DefaultMetrics.ROC_AUC,
    DefaultMetrics.PR_AUC,
    RangePrAUC(),
    RangeRocVUS(),
    RangePrVUS()
    ]

    timeeval = TimeEval(
        dm, 
        datasets,
        algorithms,
        repetitions=repetitions, 
        metrics=metrics,
        remote_config=cluster_config,
        resource_constraints=rcs,
        distributed=True,
     
    )
   
    timeeval.run()
    results = timeeval.get_results(aggregated=True)
    print(results)

if __name__ == "__main__":
            main()
