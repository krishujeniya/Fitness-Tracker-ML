import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from LearningAlgorithms_01 import ClassificationAlgorithms
from sklearn.metrics import accuracy_score,confusion_matrix

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"]=(20,5)
plt.rcParams["figure.dpi"]=100
plt.rcParams["lines.linewidth"]=2


df=pd.read_pickle("../../data/interim/03_Featured_Data.pkl")
df_train=df.drop(["Participants","Category","Set","Duration"],axis=1)
X=df_train.drop("Label",axis=1)
y=df_train["Label"]

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.25,stratify=y)
basic_col=["Accelerometer_x","Accelerometer_y","Accelerometer_z","Gyroscope_x","Gyroscope_y","Gyroscope_z"]
square_col=["Accelerometer_r","Gyroscope_r"]
pca_col=["pca_1","pca_2","pca_3"]
time_col=[f for f in df_train.columns if "_temp_" in f]
freq_col=[f for f in df_train.columns if ("_freq" in f) or ("_pse" in f)]
cluster_col=["Cluster"]
print(len(basic_col))
print(len(square_col))
print(len(pca_col))
print(len(time_col))
print(len(freq_col))
print(len(cluster_col))

f_set_1=list(set(basic_col))
f_set_2=list(set(basic_col+square_col+pca_col))
f_set_3=list(set(f_set_2+time_col))
f_set_4=list(set(f_set_3+freq_col+cluster_col))


learner=ClassificationAlgorithms()
selected_col,ordered_col,orederd_scores=learner.forward_selection(10,X_train,y_train)


iterations=1
score_df=pd.DataFrame()
possible_feature_sets=[
    f_set_1,f_set_2,f_set_3,f_set_4,selected_col
]
feature_names=[
    "f_col_1",
    "f_col_2",
    "f_col_3",
    "f_col_4",
    "selected_col",
]
for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])

score_list=score_df.sort_values(by="accuracy",ascending=False).head()
print(score_list)

class_train_y,class_test_y,class_train_prob_y,class_test_prob_y=learner.random_forest(
    X_train[f_set_4],y_train,X_test[f_set_4],gridsearch=True
)
print("RF for f_set_4 Acc=",accuracy_score(y_test,class_test_y))


classes=class_test_prob_y.columns
cm=confusion_matrix(y_test,class_test_y,labels=classes)
print("CM",cm)

participant_df=df.drop(["Set","Category"],axis=1)
X_train=participant_df[participant_df["Participants"] !="A"].drop(["Label"],axis=1)
y_train=participant_df[participant_df["Participants"] !="A"]["Label"]

X_test=participant_df[participant_df["Participants"] !="A"].drop(["Label"],axis=1)
y_test=participant_df[participant_df["Participants"] !="A"]["Label"]

X_train=X_train.drop(["Participants"],axis=1)
X_test=X_test.drop(["Participants"],axis=1)


class_train_y,class_test_y,class_train_prob_y,class_test_prob_y=learner.random_forest(
    X_train[f_set_4],y_train,X_test[f_set_4],gridsearch=True
)
print("RF for f_set_4 with participant_df Acc=",accuracy_score(y_test,class_test_y))

classes=class_test_prob_y.columns
cm=confusion_matrix(y_test,class_test_y,labels=classes)
print("CM",cm)


class_train_y,class_test_y,class_train_prob_y,class_test_prob_y=learner.feedforward_neural_network(
    X_train[selected_col],y_train,X_test[selected_col],gridsearch=False
)

print("RF for f_set_4 with selected col Acc=",accuracy_score(y_test,class_test_y))

classes=class_test_prob_y.columns
cm=confusion_matrix(y_test,class_test_y,labels=classes)
print("CM",cm)
