# import pandas as pd
from abacofs import ABACOFeatureSelector
import time
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath('.'))
# from pathfinder.featureselector import FeatureSelector


start_time = time.time()

# ACO CALL
fs = ABACOFeatureSelector(dtype='csv', data_training_name='./rtfDataSet.csv',
                          numberAnts=5, iterations=5, n_features=25)

fs.acoFS()

# TIME STOP
stop_time = time.time()
elapsed_time = stop_time - start_time

# PRINT TESTING RESULTS
fs.printTestingResults()

print("Elapsed time:")
print("\t", elapsed_time, "seconds")
