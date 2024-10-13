
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import os
from sklearn import metrics
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
import numpy as np
import random
import copy
import pandas as pd
from scipy.stats import pearsonr
import sys
import copy
import numpy as np
from operator import itemgetter

# Specifies feature selection approaches for classification to identify the most important features.
class FeatureSelectionClassification:

    # Forward selection for classification which selects a pre-defined number of features (max_features)
    # that show the best accuracy. We assume a decision tree learning for this purpose, but
    # this can easily be changed. It return the best features.
    def forward_selection(self, max_features, X_train, y_train):
        # Start with no features.
        ordered_features = []
        ordered_scores = []
        selected_features = []
        ca = ClassificationAlgorithms()
        ce = ClassificationEvaluation()
        prev_best_perf = 0

        # Select the appropriate number of features.
        for i in range(0, max_features):
            # Determine the features left to select.
            features_left = list(set(X_train.columns) - set(selected_features))
            best_perf = 0
            best_attribute = ''

            print("Added feature{}".format(i))
            # For all features we can still select...
            for f in features_left:
                temp_selected_features = copy.deepcopy(selected_features)
                temp_selected_features.append(f)

                # Determine the accuracy of a decision tree learner if we were to add
                # the feature.
                pred_y_train, pred_y_test, prob_training_y, prob_test_y = ca.decision_tree(X_train[temp_selected_features],
                                                                                           y_train,
                                                                                           X_train[temp_selected_features],
                                                                                           gridsearch=False)
                perf = ce.accuracy(y_train, pred_y_test)

                # If the performance is better than what we have seen so far (we aim for high accuracy)
                # we set the current feature to the best feature and the same for the best performance.
                if perf > best_perf:
                    best_perf = perf
                    best_feature = f

            # We select the feature with the best performance.
            selected_features.append(best_feature)
            prev_best_perf = best_perf
            ordered_features.append(best_feature)
            ordered_scores.append(best_perf)

        return selected_features, ordered_features, ordered_scores

    # Backward selection for classification which selects a pre-defined number of features (max_features)
    # that show the best accuracy. We assume a decision tree learning for this purpose, but
    # this can easily be changed. It return the best features.
    def backward_selection(self, max_features, X_train, y_train):
        # First select all features.
        selected_features = X_train.columns.tolist()
        ca = ClassificationAlgorithms()
        ce = ClassificationEvaluation()
        for i in range(0, (len(X_train.columns) - max_features)):
            best_perf = 0
            worst_feature = ''

            # Select from the features that are still in the selection.
            for f in selected_features:
                temp_selected_features = copy.deepcopy(selected_features)
                temp_selected_features.remove(f)

                # Determine the score without the feature.
                pred_y_train, pred_y_test, prob_training_y, prob_test_y = ca.decision_tree(X_train[temp_selected_features], y_train, X_train[temp_selected_features])
                perf = ce.accuracy(y_train, pred_y_train)

                # If we score better without the feature than what we have seen so far
                # this is the worst feature.
                if perf > best_perf:
                    best_perf = perf
                    worst_feature = f

            # Remove the worst feature.
            selected_features.remove(worst_feature)
        return selected_features

# Specifies feature selection approaches for classification to identify the most important features.
class FeatureSelectionRegression:

    # Forward selection for classification which selects a pre-defined number of features (max_features)
    # that show the best accuracy. We assume a decision tree learning for this purpose, but
    # this can easily be changed. It return the best features.
    def forward_selection(self, max_features, X_train, y_train):
        ordered_features = []
        ordered_scores = []

        # Start with no features.
        selected_features = []
        ra = RegressionAlgorithms()
        re = RegressionEvaluation()
        prev_best_perf = sys.float_info.max

        # Select the appropriate number of features.
        for i in range(0, max_features):

            #Determine the features left to select.
            features_left = list(set(X_train.columns) - set(selected_features))
            best_perf = sys.float_info.max
            best_feature = ''

            # For all features we can still select...
            for f in features_left:
                temp_selected_features = copy.deepcopy(selected_features)
                temp_selected_features.append(f)

                # Determine the mse of a decision tree learner if we were to add
                # the feature.
                pred_y_train, pred_y_test = ra.decision_tree(X_train[temp_selected_features], y_train, X_train[temp_selected_features])
                perf = re.mean_squared_error(y_train, pred_y_train)

                # If the performance is better than what we have seen so far (we aim for low mse)
                # we set the current feature to the best feature and the same for the best performance.
                if perf < best_perf:
                    best_perf = perf
                    best_feature = f
            # We select the feature with the best performance.
            selected_features.append(best_feature)
            prev_best_perf = best_perf
            ordered_features.append(best_feature)
            ordered_scores.append(best_perf)
        return selected_features, ordered_features, ordered_scores

    # Backward selection for classification which selects a pre-defined number of features (max_features)
    # that show the best accuracy. We assume a decision tree learning for this purpose, but
    # this can easily be changed. It return the best features.
    def backward_selection(self, max_features, X_train, y_train):

        # First select all features.
        selected_features = X_train.columns.tolist()
        ra = RegressionAlgorithms()
        re = RegressionEvaluation()

        # Select from the features that are still in the selection.
        for i in range(0, (len(X_train.columns) - max_features)):
            best_perf = sys.float_info.max
            worst_feature = ''
            for f in selected_features:
                temp_selected_features = copy.deepcopy(selected_features)
                temp_selected_features.remove(f)

                # Determine the score without the feature.
                pred_y_train, pred_y_test = ra.decision_tree(X_train[temp_selected_features], y_train, X_train[temp_selected_features])
                perf = re.mean_squared_error(y_train, pred_y_train)
                # If we score better (i.e. a lower mse) without the feature than what we have seen so far
                # this is the worst feature.
                if perf < best_perf:
                    best_perf = perf
                    worst_feature = f
            # Remove the worst feature.
            selected_features.remove(worst_feature)
        return selected_features

    # Select features based upon the correlation through the Pearson coefficient.
    # It return the max_features best features.
    def pearson_selection(self, max_features, X_train, y_train):
        correlations = []
        full_columns_and_corr = []
        abs_columns_and_corr = []

        # Compute the absolute correlations per column.
        for i in range(0, len(X_train.columns)):
            corr, p = pearsonr(X_train[X_train.columns[i]], y_train)
            correlations.append(abs(corr))
            if np.isfinite(corr):
                full_columns_and_corr.append((X_train.columns[i], corr))
                abs_columns_and_corr.append((X_train.columns[i], abs(corr)))

        sorted_attributes = sorted(abs_columns_and_corr,key=itemgetter(1), reverse=True)
        res_list = [x[0] for x in sorted_attributes[0:max_features]]

        # And return the most correlated ones.
        return res_list, sorted(full_columns_and_corr,key=itemgetter(1), reverse=True)
    

# This class creates datasets that can be used by the learning algorithms. Up till now we have
# assumed binary columns for each class, we will for instance introduce approaches to create
# a single categorical attribute.
class PrepareDatasetForLearning:

    default_label = 'undefined'
    class_col = 'class'
    person_col = 'person'

    # This function creates a single class column based on a set of binary class columns.
    # it essentially merges them. It removes the old label columns.
    def assign_label(self, dataset, class_labels):
        # Find which columns are relevant based on the possibly partial class_label
        # specification.
        labels = []
        for i in range(0, len(class_labels)):
            labels.extend([name for name in list(dataset.columns) if class_labels[i] == name[0:len(class_labels[i])]])

        # Determine how many class values are label as 'true' in our class columns.
        sum_values = dataset[labels].sum(axis=1)
        # Create a new 'class' column and set the value to the default class.
        dataset['class'] = self.default_label
        for i in range(0, len(dataset.index)):
            # If we have exactly one true class column, we can assign that value,
            # otherwise we keep the default class.
            if sum_values[i] == 1:
                dataset.iloc[i, dataset.columns.get_loc(self.class_col)] = dataset[labels].iloc[i].idxmax(axis=1)
        # And remove our old binary columns.
        dataset = dataset.drop(labels, axis=1)
        return dataset

    # Split a dataset of a single person for a classificaiton problem with the the specified class columns class_labels.
    # We can have multiple targets if we want. It assumes a list in 'class_labels'
    # If 'like' is specified in matching, we will merge the columns that contain the class_labels into a single
    # columns. We can select a filter for rows where we are unable to identifty a unique
    # class and we can select whether we have a temporal dataset or not. In the former, we will select the first
    # training_frac of the data for training and the last 1-training_frac for testing. Otherwise, we select points randomly.
    # We return a training set, the labels of the training set, and the same for a test set. We can set the random seed
    # to make the split reproducible.
    def split_single_dataset_classification(self, dataset, class_labels, matching, training_frac, filter=True, temporal=False, random_state=0):
        # Create a single class column if we have the 'like' option.
        if matching == 'like':
            dataset = self.assign_label(dataset, class_labels)
            class_labels = self.class_col
        elif len(class_labels) == 1:
            class_labels = class_labels[0]

        # Filer NaN is desired and those for which we cannot determine the class should be removed.
        if filter:
            dataset = dataset.dropna()
            dataset = dataset[dataset['class'] != self.default_label]

        # The features are the ones not in the class label.
        features = [dataset.columns.get_loc(x) for x in dataset.columns if x not in class_labels]
        class_label_indices = [dataset.columns.get_loc(x) for x in dataset.columns if x in class_labels]

        # For temporal data, we select the desired fraction of training data from the first part
        # and use the rest as test set.
        if temporal:
            end_training_set = int(training_frac * len(dataset.index))
            training_set_X = dataset.iloc[0:end_training_set, features]
            training_set_y = dataset.iloc[0:end_training_set, class_label_indices]
            test_set_X = dataset.iloc[end_training_set:len(dataset.index), features]
            test_set_y = dataset.iloc[end_training_set:len(dataset.index), class_label_indices]
        # For non temporal data we use a standard function to randomly split the dataset.
        else:
            training_set_X, test_set_X, training_set_y, test_set_y = train_test_split(dataset.iloc[:,features],
                                                                                      dataset.iloc[:,class_label_indices], test_size=(1-training_frac), stratify=dataset.iloc[:,class_label_indices], random_state=random_state)
        return training_set_X, test_set_X, training_set_y, test_set_y

    def split_single_dataset_regression_by_time(self, dataset, target, start_training, end_training, end_test):
        training_instances = dataset[start_training:end_training]
        test_instances = dataset[end_training:end_test]
        train_y = copy.deepcopy(training_instances[target])
        test_y = copy.deepcopy(test_instances[target])
        train_X = training_instances
        del train_X[target]
        test_X = test_instances
        del test_X[target]
        return train_X, test_X, train_y, test_y


    # Split a dataset of a single person for a regression with the specified targets. We can
    # have multiple targets if we want. It assumes a list in 'targets'
    # We can select whether we have a temporal dataset or not. In the former, we will select the first
    # training_frac of the data for training and the last 1-training_frac for testing. Otherwise, we select points randomly.
    # We return a training set, the labels of the training set, and the same for a test set. We can set the random seed
    # to make the split reproducible.
    def split_single_dataset_regression(self, dataset, targets, training_frac, filter=False, temporal=False, random_state=0):
        # We just temporarily change some attribute values associated with the classification algorithm
        # and change them for numerical values. We then simply apply the classification variant of the
        # function.
        temp_default_label = self.default_label
        self.default_label = np.nan
        training_set_X, test_set_X, training_set_y, test_set_y = self.split_single_dataset_classification(dataset, targets, 'exact', training_frac, filter=filter, temporal=temporal, random_state=random_state)
        self.default_label = temp_default_label
        return training_set_X, test_set_X, training_set_y, test_set_y

    # If we have multiple overlapping indices (e.g. user 1 and user 2 have the same time stamps) our
    # series cannot me merged properly, therefore we can create a new index.
    def update_set(self, source_set, addition):
        if source_set is None:
            return addition
        else:
            # Check if the index is unique. If not, create a new index.
            if len(set(source_set.index) & set(addition.index)) > 0:
                return source_set.append(addition).reset_index(drop=True)
            else:
                return source_set.append(addition)

    # If we have multiple datasets representing different users and want to perform classification,
    # we do the same as we have seen for the single dataset
    # case. However, now we can in addition select what we would like to predict: do we want to perform well for an unknown
    # use (unknown_user=True) or for unseen data over all users. In the former, it return a training set containing
    # all data of training_frac users and test data for the remaining users. If the later, it return the training_frac
    # data of each user as a training set, and 1-training_frac data as a test set.
    def split_multiple_datasets_classification(self, datasets, class_labels, matching, training_frac, filter=False, temporal=False, unknown_users=False, random_state=0):
        training_set_X = None
        training_set_y = None
        test_set_X = None
        test_set_y = None

        # If we want to learn to predict well for unknown users.
        if unknown_users:
            # Shuffle the users we have.
            random.seed(random_state)
            indices = range(0, len(datasets))
            random.shuffle(indices)
            training_len = int(training_frac * len(datasets))

            # And select the data of the first fraction training_frac of users as the training set and the data of
            # the remaining users as test set.
            for i in range(0, training_len):
                # We use the single dataset function for classification and add it to the training data
                training_set_X_person, test_set_X_person, training_set_y_person, test_set_y_person = self.split_single_dataset_classification(datasets[indices[i]], class_labels, matching,
                                                                                                                                              1, filter=filter, temporal=temporal, random_state=random_state)
                # We add a person column.
                training_set_X_person[self.person_col] = indices[i]
                training_set_X = self.update_set(training_set_X, training_set_X_person)
                training_set_y = self.update_set(training_set_y, training_set_y_person)

            for j in range(training_len, len(datasets)):
                # We use the single dataset function for classification and add it to the test data
                training_set_X_person, test_set_X_person, training_set_y_person, test_set_y_person = self.split_single_dataset_classification(datasets[indices[j]], class_labels, matching,
                                                                                                                                              1, filter=filter, temporal=temporal, random_state=random_state)
                # We add a person column.
                training_set_X_person[self.person_col] = indices[j]
                test_set_X = self.update_set(test_set_X, training_set_X_person)
                test_set_y = self.update_set(test_set_y, training_set_y_person)
        else:
            init = True
            # Otherwise we split each dataset individually in a training and test set and add them.
            for i in range(0, len(datasets)):
                training_set_X_person, test_set_X_person, training_set_y_person, test_set_y_person = self.split_single_dataset_classification(datasets[i], class_labels, matching,
                                                                                                                                              training_frac, filter=filter, temporal=temporal, random_state=random_state)
                # We add a person column.
                training_set_X_person[self.person_col] = i
                test_set_X_person[self.person_col] = i
                training_set_X = self.update_set(training_set_X, training_set_X_person)
                training_set_y = self.update_set(training_set_y, training_set_y_person)
                test_set_X = self.update_set(test_set_X, test_set_X_person)
                test_set_y = self.update_set(test_set_y, test_set_y_person)
        return training_set_X, test_set_X, training_set_y, test_set_y

    # If we have multiple datasets representing different users and want to perform regression,
    # we do the same as we have seen for the single dataset
    # case. However, now we can in addition select what we would like to predict: do we want to perform well for an unknown
    # use (unknown_user=True) or for unseen data over all users. In the former, it return a training set containing
    # all data of training_frac users and test data for the remaining users. If the later, it return the training_frac
    # data of each user as a training set, and 1-training_frac data as a test set.
    def split_multiple_datasets_regression(self, datasets, targets, training_frac, filter=False, temporal=False, unknown_users=False, random_state=0):
        # We just temporarily change some attribute values associated with the regression algorithm
        # and change them for numerical values. We then simply apply the classification variant of the
        # function.
        temp_default_label = self.default_label
        self.default_label = np.nan
        training_set_X, test_set_X, training_set_y, test_set_y = self.split_multiple_datasets_classification(datasets, targets, 'exact', training_frac, filter=filter, temporal=temporal, unknown_users=unknown_users, random_state=random_state)
        self.default_label = temp_default_label
        return training_set_X, test_set_X, training_set_y, test_set_y

# Class for evaluation metrics of classification problems.
class ClassificationEvaluation:

    # Returns the accuracy given the true and predicted values.
    def accuracy(self, y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)

    # Returns the precision given the true and predicted values.
    # Note that it returns the precision per class.
    def precision(self, y_true, y_pred):
        return metrics.precision_score(y_true, y_pred, average=None)

    # Returns the recall given the true and predicted values.
    # Note that it returns the recall per class.
    def recall(self, y_true, y_pred):
        return metrics.recall_score(y_true, y_pred, average=None)

    # Returns the f1 given the true and predicted values.
    # Note that it returns the recall per class.
    def f1(self, y_true, y_pred):
        return metrics.f1_score(y_true, y_pred, average=None)

    # Returns the area under the curve given the true and predicted values.
    # Note: we expect a binary classification problem here(!)
    def auc(self, y_true, y_pred_prob):
        return metrics.roc_auc_score(y_true, y_pred_prob)

    # Returns the confusion matrix given the true and predicted values.
    def confusion_matrix(self, y_true, y_pred, labels):
        return metrics.confusion_matrix(y_true, y_pred, labels=labels)

# Class for evaluation metrics of regression problems.
class RegressionEvaluation:

    # Returns the mean squared error between the true and predicted values.
    def mean_squared_error(self, y_true, y_pred):
        return metrics.mean_squared_error(y_true, y_pred)

    # Returns the mean squared error between the true and predicted values.
    def mean_squared_error_with_std(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        errors = np.square(y_true-y_pred)
        mse = errors.mean()
        std = errors.std()
        return mse.mean(), std.mean()

    # Returns the mean absolute error between the true and predicted values.
    def mean_absolute_error(self, y_true, y_pred):
        return metrics.mean_absolute_error(y_true, y_pred)

    # Return the mean absolute error between the true and predicted values
    # as well as its standard deviation.
    def mean_absolute_error_with_std(self, y_true, y_pred):
        errors = np.absolute((y_pred - y_true))
        return errors.mean(), errors.std()

class ClassificationAlgorithms:

    # Apply a neural network for classification upon the training data (with the specified composition of
    # hidden layers and number of iterations), and use the created network to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    # To improve the speed, one can use a CV of 3 only to make it faster
    # Furthermore, you decrease the number of iteration and increase the learning rate, i.e. 0.001 and use 'adam' as a solver
    # Include n_jobs in the GridSearchCV function and set it to -1 to use all processors which could also increase the speed
    def feedforward_neural_network(self, train_X, train_y, test_X, hidden_layer_sizes=(100,), max_iter=500, activation='logistic', alpha=0.0001, learning_rate='adaptive', gridsearch=True, print_model_details=False):


        if gridsearch:
            # With the current parameters for max_iter and Python 3 packages convergence is not always reached, therefore increased +1000.
            tuned_parameters = [{'hidden_layer_sizes': [(5,), (10,), (25,), (100,), (100,5,), (100,10,),], 'activation': [activation],
                                 'learning_rate': [learning_rate], 'max_iter': [2000, 3000], 'alpha': [alpha]}]
            nn = GridSearchCV(MLPClassifier(), tuned_parameters, cv=5, scoring='accuracy')
        else:
            # Create the model
            nn = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter, learning_rate=learning_rate, alpha=alpha, random_state=42)

        # Fit the model
        nn.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(nn.best_params_)

        if gridsearch:
            nn = nn.best_estimator_

        # Apply the model
        pred_prob_training_y = nn.predict_proba(train_X)
        pred_prob_test_y = nn.predict_proba(test_X)
        pred_training_y = nn.predict(train_X)
        pred_test_y = nn.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=nn.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=nn.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a support vector machine for classification upon the training data (with the specified value for
    # C, epsilon and the kernel function), and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    # To improve the speed, one can use a CV of 3 only to make it faster
    # Include n_jobs in the GridSearchCV function and set it to -1 to use all processors which could also increase the speed
    def support_vector_machine_with_kernel(self, train_X, train_y, test_X, C=1,  kernel='rbf', gamma=1e-3, gridsearch=True, print_model_details=False):
        # Create the model
        if gridsearch:
            tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100]}]
            svm = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5, scoring='accuracy')
        else:
            svm = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, cache_size=7000)

        # Fit the model
        svm.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(svm.best_params_)

        if gridsearch:
            svm = svm.best_estimator_

        # Apply the model
        pred_prob_training_y = svm.predict_proba(train_X)
        pred_prob_test_y = svm.predict_proba(test_X)
        pred_training_y = svm.predict(train_X)
        pred_test_y = svm.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=svm.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=svm.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a support vector machine for classification upon the training data (with the specified value for
    # C, epsilon and the kernel function), and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    # To improve the speed, one can use a CV of 3 only to make it faster and use fewer iterations
    def support_vector_machine_without_kernel(self, train_X, train_y, test_X, C=1, tol=1e-3, max_iter=1000, gridsearch=True, print_model_details=False):
        # Create the model
        if gridsearch:
            tuned_parameters = [{'max_iter': [1000, 2000], 'tol': [1e-3, 1e-4],
                         'C': [1, 10, 100]}]
            svm = GridSearchCV(LinearSVC(), tuned_parameters, cv=5, scoring='accuracy')
        else:
            svm = LinearSVC(C=C, tol=tol, max_iter=max_iter)

        # Fit the model
        svm.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(svm.best_params_)

        if gridsearch:
            svm = svm.best_estimator_

        # Apply the model

        distance_training_platt = 1/(1+np.exp(svm.decision_function(train_X)))
        pred_prob_training_y = distance_training_platt / distance_training_platt.sum(axis=1)[:,None]
        distance_test_platt = 1/(1+np.exp(svm.decision_function(test_X)))
        pred_prob_test_y = distance_test_platt / distance_test_platt.sum(axis=1)[:,None]
        pred_training_y = svm.predict(train_X)
        pred_test_y = svm.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=svm.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=svm.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y


    # Apply a nearest neighbor approach for classification upon the training data (with the specified value for
    # k), and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    # Again, use CV of 3 which will increase the speed of your model
    # Also, usage of n_jobs=-1 could help to increase the speed
    def k_nearest_neighbor(self, train_X, train_y, test_X, n_neighbors=5, gridsearch=True, print_model_details=False):
        # Create the model
        if gridsearch:
            tuned_parameters = [{'n_neighbors': [1, 2, 5, 10]}]
            knn = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5, scoring='accuracy')
        else:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)

        # Fit the model
        knn.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(knn.best_params_)

        if gridsearch:
            knn = knn.best_estimator_

        # Apply the model
        pred_prob_training_y = knn.predict_proba(train_X)
        pred_prob_test_y = knn.predict_proba(test_X)
        pred_training_y = knn.predict(train_X)
        pred_test_y = knn.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=knn.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=knn.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a decision tree approach for classification upon the training data (with the specified value for
    # the minimum samples in the leaf, and the export path and files if print_model_details=True)
    # and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    # Again, use CV of 3 which will increase the speed of your model
    # Also, usage of n_jobs in GridSearchCV could help to increase the speed
    def decision_tree(self, train_X, train_y, test_X, min_samples_leaf=50, criterion='gini', print_model_details=False, export_tree_path='./figures/crowdsignals_ch7_classification/', export_tree_name='tree.dot', gridsearch=True):
        # Create the model
        if gridsearch:
            tuned_parameters = [{'min_samples_leaf': [2, 10, 50, 100, 200],
                                 'criterion':['gini', 'entropy']}]
            dtree = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=5, scoring='accuracy')
        else:
            dtree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, criterion=criterion)

        # Fit the model

        dtree.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(dtree.best_params_)

        if gridsearch:
            dtree = dtree.best_estimator_

        # Apply the model
        pred_prob_training_y = dtree.predict_proba(train_X)
        pred_prob_test_y = dtree.predict_proba(test_X)
        pred_training_y = dtree.predict(train_X)
        pred_test_y = dtree.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=dtree.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=dtree.classes_)

        if print_model_details:
            ordered_indices = [i[0] for i in sorted(enumerate(dtree.feature_importances_), key=lambda x:x[1], reverse=True)]
            print('Feature importance decision tree:')
            for i in range(0, len(dtree.feature_importances_)):
                print(train_X.columns[ordered_indices[i]], end='')
                print(' & ', end='')
                print(dtree.feature_importances_[ordered_indices[i]])
            if not (os.path.exists(export_tree_path)):
                os.makedirs(str(export_tree_path))
            tree.export_graphviz(dtree, out_file=str(export_tree_path) + '/' + export_tree_name, feature_names=train_X.columns, class_names=dtree.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a naive bayes approach for classification upon the training data
    # and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def naive_bayes(self, train_X, train_y, test_X):
        # Create the model
        nb = GaussianNB()
        
        train_y = train_y.values.ravel()
        # Fit the model
        nb.fit(train_X, train_y)

        # Apply the model
        pred_prob_training_y = nb.predict_proba(train_X)
        pred_prob_test_y = nb.predict_proba(test_X)
        pred_training_y = nb.predict(train_X)
        pred_test_y = nb.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=nb.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=nb.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a random forest approach for classification upon the training data (with the specified value for
    # the minimum samples in the leaf, the number of trees, and if we should print some of the details of the
    # model print_model_details=True) and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    # Use CV of 3 to make things faster
    # Use n_jobs = -1 which will make use of all of your processors. This could speed up also the calculation
    def random_forest(self, train_X, train_y, test_X, n_estimators=10, min_samples_leaf=5, criterion='gini', print_model_details=False, gridsearch=True):

        if gridsearch:
            tuned_parameters = [{'min_samples_leaf': [2, 10, 50, 100, 200],
                                 'n_estimators':[10, 50, 100],
                                 'criterion':['gini', 'entropy']}]
            rf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='accuracy')
        else:
            rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, criterion=criterion)

        # Fit the model

        rf.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(rf.best_params_)

        if gridsearch:
            rf = rf.best_estimator_

        pred_prob_training_y = rf.predict_proba(train_X)
        pred_prob_test_y = rf.predict_proba(test_X)
        pred_training_y = rf.predict(train_X)
        pred_test_y = rf.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=rf.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=rf.classes_)

        if print_model_details:
            ordered_indices = [i[0] for i in sorted(enumerate(rf.feature_importances_), key=lambda x:x[1], reverse=True)]
            print('Feature importance random forest:')
            for i in range(0, len(rf.feature_importances_)):
                print(train_X.columns[ordered_indices[i]], end='')
                print(' & ', end='')
                print(rf.feature_importances_[ordered_indices[i]])

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

class RegressionAlgorithms:

    # Apply a neural network for regression upon the training data (with the specified composition of
    # hidden layers and number of iterations), and use the created network to predict the outcome for both the
    # test and training set. It returns the categorical numerical predictions for the training and test set.
    # Use CV of 3 to make things faster and might be already sufficient
    def feedforward_neural_network(self, train_X, train_y, test_X, hidden_layer_sizes=(100,), max_iter=500, activation='identity', learning_rate='adaptive', gridsearch=True, print_model_details=False):
        if gridsearch:
            # With the current parameters for max_iter and Python 3 packages convergence is not always reached, therefore increased +1000.
            tuned_parameters = [{'hidden_layer_sizes': [(5,), (10,), (25,), (100,), (100,5,), (100,10,),], 'activation': ['identity'],
                                 'learning_rate': ['adaptive'], 'max_iter': [4000, 10000]}]
            nn = GridSearchCV(MLPRegressor(), tuned_parameters, cv=5, scoring='neg_mean_squared_error')
        else:
            # Create the model
            nn = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter, learning_rate=learning_rate)

        # Fit the model
        nn.fit(train_X, train_y)

        if gridsearch and print_model_details:
            print(nn.best_params_)

        if gridsearch:
            nn = nn.best_estimator_

        # Apply the model
        pred_training_y = nn.predict(train_X)
        pred_test_y = nn.predict(test_X)

        return pred_training_y, pred_test_y

    # Apply a support vector machine with a given kernel function for regression upon the training data (with the specified value for
    # C, gamma and the kernel function), and use the created model to predict the outcome for both the
    # test and training set. It returns the predictions for the training and test set.
    # Use CV of 3 to make things faster and might be already sufficient
    # This method is rather slow and its fit time complexity is more than quadratic with the number of samples which makes scaling hard
    def support_vector_regression_with_kernel(self, train_X, train_y, test_X, kernel='rbf', C=1, gamma=1e-3, gridsearch=True, print_model_details=False):
        if gridsearch:
            tuned_parameters = [{'kernel': ['rbf', 'poly'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100]}]
            svr = GridSearchCV(SVR(), tuned_parameters, cv=5, scoring='neg_mean_squared_error')
        else:
            # Create the model
            svr = SVR(C=C, kernel='rbf', gamma=gamma)

        # Fit the model
        svr.fit(train_X, train_y)

        if gridsearch and print_model_details:
            print(svr.best_params_)

        if gridsearch:
            svr = svr.best_estimator_

        # Apply the model
        pred_training_y = svr.predict(train_X)
        pred_test_y = svr.predict(test_X)

        return pred_training_y, pred_test_y

    # Apply a support vector machine without a complex kernel function for regression upon the training data (with the specified value for
    # C, tolerance and max iterations), and use the created model to predict the outcome for both the
    # test and training set. It returns the predictions for the training and test set.
    # Use CV of 3 to make things faster and might be already sufficient
    def support_vector_regression_without_kernel(self, train_X, train_y, test_X, C=1, tol=1e-3, max_iter=1000, gridsearch=True, print_model_details=False):
        if gridsearch:
            # With the current parameters for max_iter and Python 3 packages convergence is not always reached, with increased iterations/tolerance often still fails to converge.
            tuned_parameters = [{'max_iter': [1000, 2000], 'tol': [1e-3, 1e-4],
                         'C': [1, 10, 100]}]
            svr = GridSearchCV(LinearSVR(), tuned_parameters, cv=5, scoring='neg_mean_squared_error')
        else:
            # Create the model
            svr = LinearSVR(C=C, tol=tol, max_iter=max_iter)

        # Fit the model
        svr.fit(train_X, train_y)

        if gridsearch and print_model_details:
            print(svr.best_params_)

        if gridsearch:
            svr = svr.best_estimator_

        # Apply the model
        pred_training_y = svr.predict(train_X)
        pred_test_y = svr.predict(test_X)

        return pred_training_y, pred_test_y

    # Apply a nearest neighbor approach for regression upon the training data (with the specified value for
    # k), and use the created model to predict the outcome for both the
    # test and training set. It returns the predictions for the training and test set.
    # Use CV of 3 to make things faster
    # Use n_jobs = -1 which will make use of all of your processors. This could speed up also the calculation
    def k_nearest_neighbor(self, train_X, train_y, test_X, n_neighbors=5, gridsearch=True, print_model_details=False):
        # Create the model
        if gridsearch:
            tuned_parameters = [{'n_neighbors': [1, 2, 5, 10]}]
            knn = GridSearchCV(KNeighborsRegressor(), tuned_parameters, cv=5, scoring='neg_mean_squared_error')
        else:
            # Create the model
            knn = KNeighborsRegressor(n_neighbors=n_neighbors)

        # Fit the model
        knn.fit(train_X, train_y)

        if gridsearch and print_model_details:
            print(knn.best_params_)

        if gridsearch:
            knn = knn.best_estimator_

        # Apply the model
        pred_training_y = knn.predict(train_X)
        pred_test_y = knn.predict(test_X)

        return pred_training_y, pred_test_y

    # Apply a decision tree approach for regression upon the training data (with the specified value for
    # the minimum samples in the leaf, and the export path and files if print_model_details=True)
    # and use the created model to predict the outcome for both the
    # test and training set. It returns the predictions for the training and test set.
    # Use CV of 3 to make things faster and CV of 3 might be already sufficient enough
    def decision_tree(self, train_X, train_y, test_X, min_samples_leaf=50, criterion='mse', print_model_details=False, export_tree_path='./figures/crowdsignals_ch7_regression/', export_tree_name='tree.dot', gridsearch=True):
        # Create the model
        if gridsearch:
            tuned_parameters = [{'min_samples_leaf': [2, 10, 50, 100, 200],
                                 'criterion':['mse']}]
            dtree = GridSearchCV(DecisionTreeRegressor(), tuned_parameters, cv=5, scoring='neg_mean_squared_error')
        else:
            # Create the model
            dtree = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf, criterion=criterion)

        # Fit the model
        dtree.fit(train_X, train_y)

        if gridsearch and print_model_details:
            print(dtree.best_params_)

        if gridsearch:
            dtree = dtree.best_estimator_

        # Apply the model
        pred_training_y = dtree.predict(train_X)
        pred_test_y = dtree.predict(test_X)

        if print_model_details:
            print('Feature importance decision tree:')
            ordered_indices = [i[0] for i in sorted(enumerate(dtree.feature_importances_), key=lambda x:x[1], reverse=True)]
            for i in range(0, len(dtree.feature_importances_)):
                print(train_X.columns[ordered_indices[i]], end='')
                print(' & ', end='')
                print(dtree.feature_importances_[ordered_indices[i]])
            if not (os.path.exists(export_tree_path)):
                os.makedirs(str(export_tree_path))
            tree.export_graphviz(dtree, out_file=str(export_tree_path) + '/' + export_tree_name, feature_names=train_X.columns, class_names=dtree.classes_)

        return pred_training_y, pred_test_y

    # Apply a random forest approach for regression upon the training data (with the specified value for
    # the minimum samples in the leaf, the number of trees, and if we should print some of the details of the
    # model print_model_details=True) and use the created model to predict the outcome for both the
    # test and training set. It returns the predictions for the training and test set.
    # Use CV of 3 to make things faster
    # Use n_jobs = -1 which will make use of all of your processors. This could speed up also the calculation

    def random_forest(self, train_X, train_y, test_X, n_estimators=10, min_samples_leaf=5, criterion='mse', print_model_details=False, gridsearch=True):

        if gridsearch:
            tuned_parameters = [{'min_samples_leaf': [2, 10, 50, 100, 200],
                                 'n_estimators':[10, 50, 100],
                                 'criterion':['mse']}]
            rf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, scoring='neg_mean_squared_error')
        else:
            # Create the model
            rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, criterion=criterion)

        # Fit the model
        rf.fit(train_X, train_y)

        if gridsearch and print_model_details:
            print(rf.best_params_)

        if gridsearch:
            rf = rf.best_estimator_

        # Apply the model
        pred_training_y = rf.predict(train_X)
        pred_test_y = rf.predict(test_X)

        if print_model_details:
            print('Feature importance random forest:')
            ordered_indices = [i[0] for i in sorted(enumerate(rf.feature_importances_), key=lambda x:x[1], reverse=True)]

            for i in range(0, len(rf.feature_importances_)):
                print(train_X.columns[ordered_indices[i]], end='')
                print(' & ', end='')
                print(rf.feature_importances_[ordered_indices[i]])

        return pred_training_y, pred_test_y
