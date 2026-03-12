"""
Center of Pressure (CoP) Classification for Balance Tasks

Author: Negar Rahimi

- added a main() entry point
- added optional-import guards for SHAP, LIME, and Keras
- updated deprecated tree-model settings for recent scikit-learn versions

Notes
-----
- Commented blocks were intentionally preserved when they may be useful for
  optional analyses, hyperparameter tuning, plotting, or interpretability.
- Readers can uncomment those lines if they want to run those optional steps.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import seaborn as sns
import statsmodels.api as sm
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    from keras.layers import LSTM, Dense
    from keras.models import Sequential
except Exception:
    Sequential = None
    LSTM = None
    Dense = None

try:
    import shap
except Exception:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None

try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None


DATA_FILE = "data_CWT_0102_Mohammad_time_freq"
FEATURE_NAMES_6 = [
    "X [0.16-0.23]",
    "X [0.15-0.21]",
    "X [0.14-0.20]",
    "Y [0.16-0.23]",
    "Y [0.15-0.21]",
    "Y [0.14-0.20]",
]


def _load_mat_file(path_stem: str):
    """Load a MATLAB file, trying both the provided name and a .mat suffix."""
    candidate_paths = [Path(path_stem), Path(f"{path_stem}.mat")]
    for candidate in candidate_paths:
        if candidate.exists():
            return loadmat(candidate)
    raise FileNotFoundError(
        f"Could not find '{path_stem}' or '{path_stem}.mat' in the working directory."
    )


def main():
    # 1. Data loading and train/test preparation
    mat_data = _load_mat_file(DATA_FILE)
    # mat_data = h5py.File('data_CWT_1T.mat')
    data = mat_data["data"]
    # data = np.transpose(data)
    print("Shape:", data.shape)
    # print("Data type:", data.dtype)

    # df = pd.DataFrame(data)
    # data = df[~df[df.columns[-1]].isin([3, 4])]
    # data = data.to_numpy()

    subj = data[:0]
    # X = data[:,1:7]
    X = np.hstack((data[:, 59:62], data[:, 120:123]))
    y = data[:, 123]
    # print(y)

    # df = pd.DataFrame(X)
    # correlation_matrix = df.corr().abs()
    # mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    # masked_correlation_matrix = correlation_matrix.mask(mask)
    # threshold = 0.5
    # high_correlation_features = [
    #     column
    #     for column in masked_correlation_matrix.columns
    #     if any(masked_correlation_matrix[column] > threshold)
    # ]
    # X = df.drop(high_correlation_features, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(X.shape)

    # Variance Inflation Factor (VIF)
    df = pd.DataFrame(X)
    df = sm.add_constant(df)
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    print(vif_data)

    # 2. KNN modeling and interpretation

    # KNN hyperparameter tuning (optional)
    # param_grid = {
    #     'n_neighbors': [3, 5, 7],
    #     'weights': ['uniform', 'distance'],
    #     'metric': ['euclidean', 'manhattan', 'Minkowski'],
    #     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    # }
    #
    # clf = KNeighborsClassifier()
    # grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
    # grid_search.fit(X_train, y_train)
    # best_params = grid_search.best_params_
    # best_clf = KNeighborsClassifier(**best_params)
    # best_clf.fit(X_train, y_train)
    #
    # accuracy = best_clf.score(X_test, y_test)
    # print("Best parameters found: ", grid_search.best_params_)
    # print("Best Accuracy:", accuracy)

    # knn_classifier = KNeighborsClassifier(algorithm='auto', metric='euclidean', n_neighbors=3, weights='uniform')
    knn_classifier = KNeighborsClassifier(
        algorithm="auto", metric="euclidean", n_neighbors=3, weights="distance"
    )
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy * 100)

    # labels = ['Baseline', 'C TENS', 'B TENS']
    # labels = ['FiEO', 'FiEC', 'FoEO', 'FoEC']
    # cm = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False, xticklabels=labels, yticklabels=labels)
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.title('Confusion Matrix')
    # plt.show()

    # KNN classification
    knn_classifier = KNeighborsClassifier(
        algorithm="auto", metric="euclidean", n_neighbors=3, weights="distance"
    )
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    groups = data[:, 0]
    cv_scores = cross_val_score(knn_classifier, X, y, groups=groups, cv=skf, scoring="accuracy")
    mean_accuracy = np.mean(cv_scores)
    print("Mean cross-validation accuracy: ", mean_accuracy * 100)

    plt.plot(
        range(1, len(cv_scores) + 1),
        cv_scores * 100,
        marker="o",
        linestyle="-",
        color="b",
        label="Accuracy per Fold",
    )
    plt.axhline(
        y=mean_accuracy * 100,
        color="r",
        linestyle="--",
        label=f"Mean Accuracy: {mean_accuracy * 100:.2f}%",
    )
    plt.title("Accuracy at Each Fold")
    plt.xlabel("Fold Number")
    plt.ylabel("Accuracy (%)")
    plt.xticks(range(1, len(cv_scores) + 1))
    plt.grid(True)
    plt.legend()
    plt.show()

    # 3. LIME and SHAP explanation
    df = pd.DataFrame(X, columns=FEATURE_NAMES_6)
    df["target"] = y

    X = df[FEATURE_NAMES_6]
    y = df["target"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    model = KNeighborsClassifier(
        algorithm="auto", metric="euclidean", n_neighbors=3, weights="distance"
    )

    accuracies = []
    lime_accuracies = []

    if LimeTabularExplainer is not None:
        explainer = LimeTabularExplainer(
            X.values,
            feature_names=X.columns.tolist(),
            class_names=["Baseline", "B TENS", "C TENS"],
            mode="classification",
        )

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

            lime_accuracy = 0
            for i in range(len(X_test)):
                instance = X_test.iloc[[i]]
                exp = explainer.explain_instance(
                    instance.values[0],
                    model.predict_proba,
                    num_features=len(X.columns),
                )

                lime_prediction = np.argmax(exp.predict_proba)
                if lime_prediction == y_pred[i]:
                    lime_accuracy += 1

            lime_accuracy = lime_accuracy / len(X_test)
            lime_accuracies.append(lime_accuracy)

        mean_accuracy = np.mean(accuracies)
        mean_lime_accuracy = np.mean(lime_accuracies)
        print("Mean fold accuracy:", mean_accuracy * 100)
        print("Mean LIME agreement:", mean_lime_accuracy * 100)
    else:
        print("LIME is not installed. Skipping LIME explanation section.")

    if shap is not None:
        explainer = shap.KernelExplainer(knn_classifier.predict_proba, X_train)
        # X_test = shap.sample(X_test, 6614)
        shap_values = explainer.shap_values(X_test)

        # np.save('shap_values_data_CWT_0102.npy', shap_values)
        if Path("shap_values_data_CWT_0102.npy").exists():
            shap_values = np.load("shap_values_data_CWT_0102.npy")

        if Path("shap_values_data_Time_3T_DS_Yannis.npy").exists():
            shap_values = np.load("shap_values_data_Time_3T_DS_Yannis.npy")

        print(shap_values.shape)
        print(X_test.shape)

        rcParams["font.family"] = "serif"
        rcParams["font.serif"] = ["Times New Roman"]
        rcParams["font.size"] = 18

        shap.summary_plot(shap_values[:, :, 3], X_test, feature_names=FEATURE_NAMES_6, show=False)
        plt.xlabel("SHAP Value", fontsize=14)
        plt.ylabel("Time-Frequency Features", fontsize=14)
        plt.title("Tiptoe Stance", fontsize=16)
        plt.savefig("shap_summary_TTS.png", bbox_inches="tight")
        plt.show()

        features = X_test
        shap.dependence_plot(2, shap_values[:, :, 0], X_test, interaction_index=None)
        plt.show()

        shap.decision_plot(
            explainer.expected_value[0],
            shap_values[:, :, 1],
            X_test.columns,
            ignore_warnings=True,
        )
    else:
        print("SHAP is not installed. Skipping SHAP explanation section.")

    # 4. Tree-based models

    # RF hyperparameters tuning (optional)
    # rf = RandomForestClassifier(random_state=42)
    # param_grid = {
    #     'n_estimators': [100, 200],
    #     'max_features': ['sqrt', 'log2'],
    #     'min_samples_split': [2, 5],
    #     'min_samples_leaf': [1, 2],
    #     'criterion': ['gini', 'entropy'],
    # }
    # grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2, scoring='accuracy')
    # grid_search.fit(X_train, y_train)
    # print("Best parameters found: ", grid_search.best_params_)
    # print("Best accuracy found: ", grid_search.best_score_)

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="entropy",
        n_jobs=-1,
    )

    rf.fit(X, y)
    feature_importances = rf.feature_importances_

    importances_df = pd.DataFrame(
        {
            "Feature": FEATURE_NAMES_6,
            "Importance": feature_importances,
        }
    )
    importances_df = importances_df.sort_values(by="Importance", ascending=False)

    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(10, 6))
    plt.barh(importances_df["Feature"], importances_df["Importance"])
    plt.xlabel("Importance", fontsize=18, fontname="Arial")
    plt.ylabel("Features", fontsize=18, fontname="Arial")
    plt.title("Time-Frequency Domain", fontsize=20, fontweight="bold", fontname="Arial")
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("FeatImp_TimefREQ_Mohammad.png")
    plt.show()

    # Decision Tree classification
    clf = DecisionTreeClassifier(
        random_state=42,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="entropy",
        splitter="best",
    )
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    groups = data[:, 0]
    cv_scores = cross_val_score(clf, X, y, groups=groups, cv=skf, scoring="accuracy")
    mean_accuracy = np.mean(cv_scores)
    print("Mean cross-validation accuracy: ", mean_accuracy * 100)

    # Gradient Boosting section (optional)
    # gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    # skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # groups = data[:, 0]
    # cv_scores = cross_val_score(gbm, X, y, groups=groups, cv=skf, scoring='accuracy')
    # mean_accuracy = np.mean(cv_scores)
    # print('Mean cross-validation accuracy: ', mean_accuracy * 100)

    # 5. SVM model
    svm_classifier = SVC(C=10, gamma="scale", kernel="rbf")
    scores = cross_val_score(svm_classifier, X, y, cv=5)
    print("Mean accuracy:", scores.mean() * 100)

    # 6. Summary plots
    groups = 3
    bars_per_group = 5
    bar_labels = ["Time", "CoPx", "CoPy", "CWTx", "CWTy"]
    bar_values = np.array(
        [
            [0.0588, 0.1115, 0.1411, 0.1922, 0.2107],
            [0.1105, 0.1115, 0.2220, 0.1965, 0.2067],
            [0.2228, 0.1115, 0.1112, 0.1978, 0.2053],
        ]
    )
    colors = ["orange", "orange", "orange", "green", "green"]

    bar_width = 0.15
    inter_bar_space = 0.05
    inter_group_space = 0.2

    group_locs = np.arange(groups) * (
        bars_per_group * (bar_width + inter_bar_space) + inter_group_space
    )
    bar_locs = []
    bar_names = []

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.rcParams.update(
        {"font.family": "serif", "font.serif": "Times New Roman", "font.size": 14}
    )

    for i in range(bars_per_group):
        locs = group_locs + i * (bar_width + inter_bar_space)
        ax.bar(locs, bar_values[:, i], bar_width, color=colors[i])
        bar_locs.extend(locs)
        bar_names.extend([bar_labels[i]] * groups)

    ax.set_ylabel("Average SHAP Values", fontsize=18)
    ax.set_xticks(bar_locs)
    ax.set_xticklabels(bar_names, rotation=90, ha="center", fontsize=14)

    custom_legend = [
        plt.Line2D([0], [0], color="orange", lw=4, label="Time Features"),
        plt.Line2D([0], [0], color="green", lw=4, label="Time-Frequency Features"),
    ]
    ax.legend(handles=custom_legend, loc="upper right", fontsize=12)

    plt.tight_layout()
    plt.savefig("AvgSHAP_BarPlot_ExpA.png")
    plt.show()

    groups = 2
    bars_per_group = 5
    bar_labels = ["Time", "CoPx", "CoPy", "CWTx", "CWTy"]
    bar_values = np.array(
        [
            [0.1172, 0.1578, 0.1408, 0.2443, 0.2234],
            [0.1273, 0.1292, 0.1449, 0.2063, 0.2613],
        ]
    )
    colors = ["orange", "orange", "orange", "green", "green"]

    bar_width = 0.15
    inter_bar_space = 0.05
    inter_group_space = 0.2

    group_locs = np.arange(groups) * (
        bars_per_group * (bar_width + inter_bar_space) + inter_group_space
    )
    bar_locs = []
    bar_names = []

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.rcParams.update(
        {"font.family": "serif", "font.serif": "Times New Roman", "font.size": 14}
    )

    for i in range(bars_per_group):
        locs = group_locs + i * (bar_width + inter_bar_space)
        ax.bar(locs, bar_values[:, i], bar_width, color=colors[i])
        bar_locs.extend(locs)
        bar_names.extend([bar_labels[i]] * groups)

    ax.set_ylabel("Average SHAP Values", fontsize=18)
    ax.set_xticks(bar_locs)
    ax.set_xticklabels(bar_names, rotation=90, ha="center", fontsize=14)

    custom_legend = [
        plt.Line2D([0], [0], color="orange", lw=4, label="Time Features"),
        plt.Line2D([0], [0], color="green", lw=4, label="Time-Frequency Features"),
    ]
    ax.legend(handles=custom_legend, loc="upper right", fontsize=12)

    plt.tight_layout()
    plt.savefig("AvgSHAP_BarPlot_ExpB.png")
    plt.show()


if __name__ == "__main__":
    main()
