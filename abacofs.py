from pathfinder.featureselector import FeatureSelector
import sys
import numpy as np
import pandas as pd
import scipy.io as sp
import time
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from pathfinder.ant import Ant
np.set_printoptions(threshold=sys.maxsize)

class ABACOFeatureSelector(FeatureSelector):

    def defineLUT(self):
        """Defines the Look-Up Table (LUT) for the algorithm.
        """
        time_LUT_start = time.time()

        fs = SelectKBest(score_func=mutual_info_classif, k='all')
        
        # TO DO: I want to know what is this FS     
        fs.fit(self.data_training, self.class_training)
        self.LUT = fs.scores_
        sum = np.sum(self.LUT)
        
        # TO DO: it seems like it is alredy averaging scored. We have to compare it to the class to get the score
        for i in range(len(fs.scores_)):
            self.LUT[i] = self.LUT[i]/sum
        
        time_LUT_stop = time.time()
        self.time_LUT = self.time_LUT + (time_LUT_stop - time_LUT_start)

    def redefineLUT(self, feature): 
        """Re-defines the Look-Up Table (LUT) for the algorithm.
        """
        time_LUT_start = time.time()
        
        weightprob = self.LUT[feature]
        self.LUT[feature] = 0
        mult = 1/(1-weightprob)
        self.LUT = self.LUT * mult
        
        time_LUT_stop = time.time()
        self.time_LUT = self.time_LUT + (time_LUT_stop - time_LUT_start)
        