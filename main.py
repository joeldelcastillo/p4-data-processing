# import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath('.'))
import time
from pathfinder.featureselector import FeatureSelector


start_time = time.time()

# ACO CALL
# fs = FeatureSelector(sys.argv[1], sys.argv[2], None ,int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
fs = FeatureSelector(dtype='csv',data_training_name='./rtfDataSet.csv', numberAnts=50, iterations=20, n_features=20)

fs.acoFS()

# TIME STOP
stop_time = time.time()
elapsed_time = stop_time - start_time

# PRINT TESTING RESULTS
fs.printTestingResults()

print("Elapsed time:")
print("\t", elapsed_time, "seconds")