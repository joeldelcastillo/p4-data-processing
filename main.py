# import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath('.'))
import time
# from pathfinder.featureselector import FeatureSelector
from abacofs import ABACOFeatureSelector


start_time = time.time()

# ACO CALL
fs = ABACOFeatureSelector(dtype='csv',data_training_name='./rtfDataSet.csv', numberAnts=30, iterations=10, n_features=25)

fs.acoFS()

# TIME STOP
stop_time = time.time()
elapsed_time = stop_time - start_time

# PRINT TESTING RESULTS
fs.printTestingResults()

print("Elapsed time:")
print("\t", elapsed_time, "seconds")