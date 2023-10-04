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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from pathfinder.ant import Ant
np.set_printoptions(threshold=sys.maxsize)


class ABACOFeatureSelector(FeatureSelector):

    def __init__(self, dtype="mat", data_training_name=None, class_training_name=None, numberAnts=1, iterations=1, n_features=1, data_testing_name=None, class_testing_name=None, alpha=1, beta=1, Q_constant=1, initialPheromone=1.0, evaporationRate=0.1):
        """Constructor method.
        """
        time_dataread_start = time.time()
        if dtype == "mat":
            dic_data_training = sp.loadmat(data_training_name)
            dic_class_training = sp.loadmat(class_training_name)
            self.data_training = np.array(dic_data_training["data"])
            self.class_training = np.reshape(
                np.array(dic_class_training["class"]), len(self.data_training)) - 1

            dic_data_testing = sp.loadmat(data_testing_name)
            dic_class_testing = sp.loadmat(class_testing_name)
            self.data_testing = np.array(dic_data_testing["data"])
            self.class_testing = np.reshape(
                np.array(dic_class_testing["class"]), len(self.data_testing)) - 1

            # Free dictionaries memory
            del dic_data_training
            del dic_class_training

        elif dtype == "csv":
            df = pd.read_csv(data_training_name)

            # Normalize
            scaler = MinMaxScaler()
            scaler.fit(df)
            scaled = scaler.fit_transform(df)
            scaled_df = pd.DataFrame(scaled, columns=df.columns)

            print(scaled_df.head())
            df = scaled_df.to_numpy()

            # df = df.to_numpy()
            classes = df[:, -1].astype(int)
            df = np.delete(df, -1, 1)
            self.data_training, self.data_testing, self.class_training, self.class_testing = train_test_split(
                df, classes, random_state=42)

        # print("Samples x features:", np.shape(self.dataset))

        scaler = StandardScaler().fit(self.data_training)
        self.data_training = scaler.transform(self.data_training)
        self.data_testing = scaler.transform(self.data_testing)

        self.number_ants = numberAnts
        self.ants = [Ant() for _ in range(self.number_ants)]
        self.number_features = len(self.data_training[0])
        self.iterations = iterations
        self.initial_pheromone = initialPheromone
        self.evaporation_rate = evaporationRate
        self.alpha = alpha
        self.beta = beta
        self.Q_constant = Q_constant
        self.feature_pheromone = np.full(
            self.number_features, self.initial_pheromone)
        self.unvisited_features = np.arange(self.number_features)
        self.ant_accuracy = np.zeros(self.number_ants)
        self.n_features = n_features
        if self.n_features > self.number_features:
            self.n_features = self.number_features
        #random.seed(1) ########################################################################

        time_dataread_stop = time.time()
        self.time_dataread = time_dataread_stop - time_dataread_start
        self.time_LUT = 0
        self.time_reset = 0
        self.time_localsearch = 0
        self.time_pheromonesupdate = 0

        self.all_subsets = []
        self.subset_percapita = []
        self.last_colony = []

    def defineLUT(self):
        """Defines the Look-Up Table (LUT) for the algorithm.
        """
        time_LUT_start = time.time()

        fs = SelectKBest(score_func=mutual_info_classif, k='all')

        # TO DO: I want to know what is this FS
        fs.fit(self.data_training, self.class_training)
        self.LUT = fs.scores_
        sum = np.sum(self.LUT)

        # Here is just scaling the numbers to sum 1 for probabilities.
        for i in range(len(fs.scores_)):
            self.LUT[i] = self.LUT[i]/sum

        time_LUT_stop = time.time()
        self.time_LUT = self.time_LUT + (time_LUT_stop - time_LUT_start)

    def per_capita(self, feature_subset):
        per_capita_importance = 0
        for i, feature in enumerate(feature_subset):
            per_capita_importance += self.LUT[feature]
        per_capita_importance /= len(feature_subset)
        self.subset_percapita.append((feature_subset, per_capita_importance))

    def save_last_colony_subset(self, feature_path):
        self.last_colony.append(feature_path)

    # TO DO: We have to compute the ABACO.
    # Fun fact: BACO is just an ACO with binary selection, which is true for our dataset. (I don't we we need to change that)
    def acoFS(self):
        """Compute the original ACO algorithm workflow. Firstly it resets the values of the ants (:py:meth:`featureselector.FeatureSelector.resetInitialValues`), 
        """
        self.defineLUT()

        for c in range(self.iterations):
            # TO DO: Again: We have to do search about how ABACO works. But I believe that it does not reset values.
            self.resetInitialValues()
            print("Colony", c, ":")
            ia = 0
            for ia in range(self.number_ants):
                self.antBuildSubset(ia)
                if (c == self.iterations-1):
                    self.save_last_colony_subset(self.ants[ia].feature_path)
                print("\tAnt", ia, ":")
                print("\t\tPath:", self.ants[ia].feature_path)
                print("\t\tAccuracy:", self.ant_accuracy[ia])
            self.updatePheromones()

        for subset in self.last_colony:
            self.per_capita(subset)
            #print("\t\tPheromones: \t", self.feature_pheromone)

    def printTopFive(self):
        # Sort the list of tuples based on the second element of each tuple in descending order
        sorted_tuples = sorted(
            self.subset_percapita, key=lambda x: x[1], reverse=True)

        # Get the top five tuples
        top_five_tuples = sorted_tuples[:5]

        print("Top 5 ants:")
        # Print the top five tuples
        for item in top_five_tuples:
            print("Path: ", item[0], "\t | IPC: ", item[1])

    def printTestingResults(self):
        """Function for printing the entire summary of the algorithm, including the test results.
        """
        print("The final subset of features is: ",
              self.ants[np.argmax(self.ant_accuracy)].feature_path)

        self.printTopFive()

        print("Number of features: ", len(
            self.ants[np.argmax(self.ant_accuracy)].feature_path))

        data_training_subset = self.data_training[:, self.ants[np.argmax(
            self.ant_accuracy)].feature_path]
        data_testing_subset = self.data_testing[:, self.ants[np.argmax(
            self.ant_accuracy)].feature_path]

        print("Subset of features dataset accuracy:")

        knn = KNeighborsClassifier()
        knn.fit(data_training_subset, self.class_training)
        knn_score = knn.score(data_testing_subset, self.class_testing)
        print("\t CV-Training set: ", np.max(self.ant_accuracy))
        print("\t Testing set    : ", knn_score)

        print("\t Time elapsed reading data        : ", self.time_dataread)
        print("\t Time elapsed in LUT compute      : ", self.time_LUT)
        print("\t Time elapsed reseting values     : ", self.time_reset)
        print("\t Time elapsed in local search     : ", self.time_localsearch)
        print("\t Time elapsed updating pheromones : ",
              self.time_pheromonesupdate)

        print()

        predicted_probabilities = knn.predict_proba(
            data_testing_subset)[:, 1]
        auc = roc_auc_score(self.class_testing, predicted_probabilities)

        print("TOTAL AUC FROM MODEL: ", auc)

        subset_five = sorted(
            self.subset_percapita, key=lambda x: x[1], reverse=True)
        subset_five = subset_five[:5]

        for i in subset_five:
            data_testing_subset = self.data_testing[:, i[0]]
            predicted_probabilities = knn.predict_proba(
                data_testing_subset)[:, 1]
            auc = roc_auc_score(self.class_testing, predicted_probabilities)
            print(f"AUC Score {i}:", auc)
