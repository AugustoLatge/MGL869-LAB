import os
import sys
import logging
import math
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import auc, make_scorer, precision_score, recall_score, roc_curve, RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer
from collections import Counter
import matplotlib.pyplot as plt
from pickle import dump, load
from statistics import stdev


def generate_final_project_part_2_model(current_version, recalculate_models=True, plot_images=True):
    dots_separated_current_version = ".".join(current_version.split("_"))

    base_dir = Path(os.path.realpath(__file__)).parent.parent.parent

    data_dir = base_dir / "data"

    metrics_dir = data_dir / "metrics"

    new_metrics_dir = data_dir / "new_metrics"

    output_new_metrics_dir = base_dir / "output" / "new_metrics" / "part_2"
    output_new_metrics_dir.mkdir(exist_ok=True)

    version_output_dir = output_new_metrics_dir / current_version
    version_output_dir.mkdir(exist_ok=True)

    all_metrics_path = metrics_dir / f"und_hive_all_metrics_{current_version}.csv"

    logging.basicConfig(level=logging.DEBUG)

    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    file_handler = logging.FileHandler(version_output_dir / f"logs_{current_version}.log", mode='w')
    logger = logging.getLogger()
    for handler in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(handler)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    logging.info(f"VERSION: {dots_separated_current_version}")
    logging.info("")

    if not all_metrics_path.exists():

        metrics_path = metrics_dir / f"und_hive_{current_version}.csv"

        dataset = pd.read_csv(metrics_path)
        # List of column names to be removed
        columns_to_remove = ["CCViolDensityCode", "CCViolDensityLine", "CountCCViol", "CountCCViolType",
                             "CountClassCoupledModified", "CountDeclExecutableUnit", "CountDeclFile",
                             "CountDeclMethodAll", "Cyclomatic", "PercentLackOfCohesionModified"]
        # Remove the specified columns
        dataset = dataset.drop(columns=columns_to_remove)

        filtered_dataset = dataset[dataset["Kind"] == "File"].copy()
        dataset = dataset[dataset["Kind"] != "File"]

        def calculate_value_class(dataset, column_name, mask):
            class_data = dataset.loc[mask & dataset["Kind"].str.contains("Class")]
            if class_data.empty:
                return np.nan

            if class_data[column_name].empty:
                return np.nan
            return class_data[column_name].mean()

        def calculate_value_method(dataset, column_name, specification, mask):
            method_data = dataset.loc[mask & dataset["Kind"].str.contains("Method")]
            if method_data.empty:
                return np.nan

            if "Min" in specification:
                return method_data[column_name].min()
            elif "Max" in specification:
                return method_data[column_name].max()
            else:  # Mean
                if method_data[column_name].empty:
                    return np.nan
                return method_data[column_name].mean()

        classes_metrics = ["CountClassBase", "CountClassCoupled", "CountClassDerived", "MaxInheritanceTree",
                           "PercentLackOfCohesion"]
        methods_metrics = ["CountInput", "CountOutput", "CountPath", "MaxNesting"]
        methods_specification = ["Min", "Mean", "Max"]

        file_names = filtered_dataset["Name"].apply(lambda x: Path(x).stem)
        masks = {file_name: dataset["Name"].str.contains(file_name) for file_name in file_names}

        for i, file_name in enumerate(filtered_dataset["Name"], start=1):
            logging.info(f"{i} - {file_name}")
            file_name_without_extension = Path(file_name).stem

            mask = masks[file_name_without_extension]
            if not mask.any():
                logging.info(f"{file_name} not found in dataset, skipping...")
                continue

            class_values = {col: calculate_value_class(dataset, col, mask) for col in classes_metrics}
            for col, value in class_values.items():
                filtered_dataset.loc[filtered_dataset["Name"] == file_name, col] = 0 if np.isnan(value) else value

            for col in methods_metrics:
                method_values = {spec: calculate_value_method(dataset, col, spec, mask) for spec in
                                 methods_specification}
                for spec, value in method_values.items():
                    filtered_dataset.loc[filtered_dataset["Name"] == file_name, col + spec] = 0 if np.isnan(
                        value) else value

        # Remove the specified columns
        filtered_dataset = filtered_dataset.drop(columns=methods_metrics)

        filtered_dataset.to_csv(metrics_dir / f"und_hive_all_metrics_{current_version}.csv", index=False)
    else:
        filtered_dataset = pd.read_csv(all_metrics_path)

    # Replace negative values with 0
    filtered_dataset.iloc[:, 2:] = filtered_dataset.iloc[:, 2:].clip(lower=0)

    # Add new metrics to dataset
    new_metrics_dataset = pd.read_csv(new_metrics_dir / f"New_files_metrics_{dots_separated_current_version}.csv")
    filtered_dataset[new_metrics_dataset.columns[1:]] = 0
    for i in range(len(filtered_dataset)):
        if filtered_dataset.loc[i, "Name"] in new_metrics_dataset["Name"].values:
            filtered_dataset.loc[i, new_metrics_dataset.columns[1:]] = new_metrics_dataset[
                                                                           new_metrics_dataset["Name"] ==
                                                                           filtered_dataset.loc[i, "Name"]].iloc[0, 1:]
    logging.info(f"Initial number of files in filtered_dataset: {len(filtered_dataset)}")
    logging.info(f"Number of files in new_metrics_dataset: {len(new_metrics_dataset)}")
    logging.info(
        f"Number of files of filtered_dataset not in new_metrics_dataset: {len(filtered_dataset[~filtered_dataset["Name"].isin(new_metrics_dataset["Name"].values)])}")
    filtered_dataset = filtered_dataset[filtered_dataset["Name"].isin(new_metrics_dataset["Name"].values)].reset_index(
        drop=True)
    logging.info(f"Number of files in the intersection of the two datasets: {len(filtered_dataset)}")
    logging.info("")

    # Trivial and Minor are considered as one class
    PRIORITY_DICT = {
        "None": "None",
        "Trivial": "Trivial or Minor",
        "Minor": "Trivial or Minor",
        "Major": "Major",
        "Critical": "Critical",
        "Blocker": "Blocker",
    }

    # Read the files with priorities
    files_with_priorities = pd.read_csv(new_metrics_dir / f"Priorities_{dots_separated_current_version}.csv")
    files_with_priorities = files_with_priorities.drop("key", axis=1)
    files_with_priorities = files_with_priorities.rename(columns={"filename": "Name", "priority": "Priority"})
    files_with_priorities["Name"] = files_with_priorities["Name"].apply(lambda x: Path(x).name)
    files_with_priorities = files_with_priorities[files_with_priorities["Name"].str.endswith(".java")]
    files_with_priorities = files_with_priorities.drop_duplicates()
    files_with_priorities["Name"] = files_with_priorities["Name"].apply(lambda x: Path(x).name)
    files_with_priorities["Priority"] = files_with_priorities["Priority"].apply(lambda x: PRIORITY_DICT[x])

    # Add "Priority" column
    filtered_dataset["Priority"] = "None"
    for i in range(len(filtered_dataset)):
        if filtered_dataset.loc[i, "Name"] in files_with_priorities["Name"].values:
            # If there are more than one priority for the same file name, get the highest one
            filtered_dataset.loc[i, "Priority"] = files_with_priorities[
                files_with_priorities["Name"] == filtered_dataset.loc[i, "Name"]]["Priority"].max()
    logging.info(f"Total number of .java files: {len(filtered_dataset)}")
    logging.info(
        f"Number of files without bugs (no priority): {len(filtered_dataset[filtered_dataset["Priority"] == "None"])}")
    logging.info(
        f"Number of files with Trivial or Minor priority: {len(filtered_dataset[filtered_dataset["Priority"] == "Trivial or Minor"])}")
    logging.info(f"Number of files with Major priority: {len(filtered_dataset[filtered_dataset["Priority"] == "Major"])}")
    logging.info(f"Number of files with Critical priority: {len(filtered_dataset[filtered_dataset["Priority"] == "Critical"])}")
    logging.info(f"Number of files with Blocker priority: {len(filtered_dataset[filtered_dataset["Priority"] == "Blocker"])}")
    logging.info(
        f"Total number of files with priorities: {len(filtered_dataset[filtered_dataset["Priority"].isin(["Trivial or Minor", "Major", "Critical", "Blocker"])])}")
    logging.info(f"Missing .java files in the filtered_dataset:")
    for file in files_with_priorities["Name"]:
        if file not in list(filtered_dataset["Name"]):
            logging.info(f"    {file}")
    logging.info("")

    # Save all metrics and bugs to file
    filtered_dataset.to_csv(new_metrics_dir / f"und_hive_all_metrics_and_priorities_{current_version}.csv", index=False)

    # Drop "Kind" column
    filtered_dataset = filtered_dataset.drop("Kind", axis=1)

    def divided_count_path(dataset, operation):
        """Change "CountPath" scale because numbers are too big in regard to other columns.
        The numbers of the new scale will have a maximum of 4 digits."""
        count_path_operation = f"CountPath{operation}"
        max_nb_of_digits = math.floor(math.log10(max(dataset[count_path_operation]))) + 1
        division_factor = 10 ** (max_nb_of_digits - 3)
        if division_factor == 1:
            return dataset
        dataset[count_path_operation] = dataset[count_path_operation].apply(
            lambda x: x if math.isnan(x) else int(round(x / division_factor, 0)))
        return dataset.rename(columns={count_path_operation: f"{count_path_operation}-divided-by-{division_factor:,}"})

    for operation in ["Min", "Max", "Mean"]:
        filtered_dataset = divided_count_path(filtered_dataset, operation)

    # Display initial variable columns
    initial_columns = list(filtered_dataset.columns[1:-1])
    logging.info(f"Initial variable columns: {len(initial_columns)}")
    logging.info("")

    # Drop columns with all NaN
    filtered_dataset = filtered_dataset.dropna(axis=1, how='all')
    remaining_columns = list(filtered_dataset.columns[1:-1])
    logging.info("Drop all NaN columns")
    logging.info(f"Remaining columns ({len(remaining_columns)}):")
    for column in remaining_columns:
        logging.info(f"    {column}")
    dropped_columns = [column for column in initial_columns if column not in remaining_columns]
    logging.info(f"Dropped all NaN columns ({len(dropped_columns)}):")
    for column in dropped_columns:
        logging.info(f"    {column}")
    logging.info("")

    # Check for missing values
    logging.info("Columns with missing values:")
    missing_values = filtered_dataset.iloc[:, 1:-1].isnull().sum()
    for column in missing_values.index:
        logging.info(f"    {column}: {missing_values[column]}")
    logging.info(
        f"Total rows with missing values removed: {len(filtered_dataset[~(filtered_dataset.index.isin(filtered_dataset.dropna().index))])}")
    filtered_dataset = filtered_dataset.dropna()
    logging.info(f"Total rows remaining: {len(filtered_dataset)}")
    logging.info("")

    # Remove correlated columns
    # Ref.: https://www.kaggle.com/code/prashant111/comprehensive-guide-on-feature-selection#2.6-Correlation-Matrix-with-Heatmap-
    corr_matrix = filtered_dataset.iloc[:, 1:-1].corr()

    if plot_images:
        # Create correlation heatmap
        plt.figure(figsize=(77, 75))
        plt.title(f'Correlation Heatmap version {dots_separated_current_version}')
        a = sns.heatmap(corr_matrix, square=True, annot=True, fmt='.2f', linecolor='black')
        a.set_xticklabels(a.get_xticklabels(), rotation=30)
        a.set_yticklabels(a.get_yticklabels(), rotation=30)
        plt.savefig(version_output_dir / f"correlation_heatmap_{current_version}.png")

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.7
    correlation_treshold = 0.7
    to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_treshold)]
    logging.info("Correlated columns to be dropped:")
    for column in to_drop:
        correlated_to = list(upper[upper[column].abs() > correlation_treshold].index)
        logging.info(f"    {column}, correlated to: {correlated_to}")
    logging.info("")

    # Drop correlated columns
    filtered_dataset = filtered_dataset.drop(to_drop, axis=1)

    # Checking boxplots (ref.: https://www.kaggle.com/code/marcinrutecki/gridsearchcv-kfold-cv-the-right-way)
    def boxplots_custom(filtered_dataset, columns_list, rows, cols, suptitle):
        fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(13, 50))
        fig.suptitle(suptitle, y=1, size=25)
        axs = axs.flatten()
        for i, data in enumerate(columns_list):
            sns.boxplot(data=filtered_dataset[data], orient='h', ax=axs[i])
            axs[i].set_title(data + ', skewness is: ' + str(round(filtered_dataset[data].skew(axis=0, skipna=True), 2)))

    columns_list = list(filtered_dataset.columns[1:-1])
    if plot_images:
        boxplots_custom(filtered_dataset=filtered_dataset, columns_list=columns_list,
                        rows=math.ceil(len(columns_list) / 3), cols=3,
                        suptitle='Boxplots for each variable')
        plt.tight_layout()
        plt.savefig(version_output_dir / f"boxplots_{current_version}.png")

    def IQR_method(df, n, features):
        """
        Takes a dataframe and returns an index list corresponding to the observations
        containing more than n outliers according to the Tukey IQR method.
        Ref.: https://www.kaggle.com/code/marcinrutecki/gridsearchcv-kfold-cv-the-right-way
        """
        outlier_list = []

        for column in features:
            # 1st quartile (25%)
            Q1 = np.percentile(df[column], 25)
            # 3rd quartile (75%)
            Q3 = np.percentile(df[column], 75)
            # Interquartile range (IQR)
            IQR = Q3 - Q1
            outlier_step = 1.5 * IQR
            # Determining a list of indices of outliers
            outlier_list_column = df[(df[column] < Q1 - outlier_step) | (df[column] > Q3 + outlier_step)].index
            # appending the list of outliers
            outlier_list.extend(outlier_list_column)

        # selecting observations containing more than x outliers
        outlier_list = Counter(outlier_list)
        multiple_outliers = list(k for k, v in outlier_list.items() if v > n)

        return multiple_outliers

    # Remove outliers (save the outliers to disk)
    # Adjust the `n` argument of `IQR_method` to allow more outliers to be kept, otherwise most of the files with bugs
    # where being removed
    n = 20
    logging.info("Remove outliers:")
    logging.info(f"    Initial number of rows in the filtered_dataset: {len(filtered_dataset)}")
    logging.info(
        f"    Initial number of files without bugs (no priority): {len(filtered_dataset[filtered_dataset["Priority"] == "None"])}")
    logging.info(
        f"    Initial number of files with Trivial or Minor priority: {len(filtered_dataset[filtered_dataset["Priority"] == "Trivial or Minor"])}")
    logging.info(
        f"    Initial number of files with Major priority: {len(filtered_dataset[filtered_dataset["Priority"] == "Major"])}")
    logging.info(
        f"    Initial number of files with Critical priority: {len(filtered_dataset[filtered_dataset["Priority"] == "Critical"])}")
    logging.info(
        f"    Initial number of files with Blocker priority: {len(filtered_dataset[filtered_dataset["Priority"] == "Blocker"])}")
    logging.info(
        f"    Total initial number of files with priorities: {len(filtered_dataset[filtered_dataset["Priority"].isin(["Trivial or Minor", "Major", "Critical", "Blocker"])])}")
    logging.info(f"    IQR_method n argument: {n}")
    outliers_IQR = IQR_method(filtered_dataset, n, columns_list)
    outliers = filtered_dataset.loc[outliers_IQR].reset_index(drop=True)
    logging.info(f"    Total number of outliers is: {len(outliers_IQR)}")
    # Drop outliers
    filtered_dataset = filtered_dataset.drop(outliers_IQR, axis=0).reset_index(drop=True)
    logging.info(f"    Final number of rows in the filtered_dataset: {len(filtered_dataset)}")
    logging.info(
        f"    Final number of files without bugs (no priority): {len(filtered_dataset[filtered_dataset["Priority"] == "None"])}")
    logging.info(
        f"    Final number of files with Trivial or Minor priority: {len(filtered_dataset[filtered_dataset["Priority"] == "Trivial or Minor"])}")
    logging.info(
        f"    Final number of files with Major priority: {len(filtered_dataset[filtered_dataset["Priority"] == "Major"])}")
    logging.info(
        f"    Final number of files with Critical priority: {len(filtered_dataset[filtered_dataset["Priority"] == "Critical"])}")
    logging.info(
        f"    Final number of files with Blocker priority: {len(filtered_dataset[filtered_dataset["Priority"] == "Blocker"])}")
    logging.info(
        f"    Total Final number of files with priorities: {len(filtered_dataset[filtered_dataset["Priority"].isin(["Trivial or Minor", "Major", "Critical", "Blocker"])])}")
    logging.info("")

    # Drop columns with all same value
    initial_columns = list(filtered_dataset.columns[1:-1])
    logging.info("Drop same value columns after outliers removal")
    logging.info(f"Initial columns: {len(initial_columns)}")
    number_unique = filtered_dataset.nunique()
    cols_to_drop = number_unique[number_unique == 1].index
    filtered_dataset = filtered_dataset.drop(cols_to_drop, axis=1)
    outliers_dataset = outliers.drop(cols_to_drop, axis=1)
    remaining_columns = list(filtered_dataset.columns[1:-1])
    logging.info(f"Remaining columns ({len(remaining_columns)}):")
    for column in remaining_columns:
        logging.info(f"    {column}")
    dropped_columns = [column for column in initial_columns if column not in remaining_columns]
    logging.info(f"Dropped same value columns ({len(dropped_columns)}):")
    for column in dropped_columns:
        logging.info(f"    {column}")
    logging.info("")

    # Print variables range
    logging.info("Variables range:")
    for column in filtered_dataset.columns[1:-1]:
        logging.info(
            f"    {column}: {round(min(filtered_dataset[column]), 1)} - {round(max(filtered_dataset[column]), 1)}")
    logging.info("")

    # Save preprocessed data to file
    # filtered_dataset.to_csv(version_output_dir / f"und_hive_metrics_preprocessed_{current_version}.csv", index=False)

    # Save outliers data to file
    # outliers.to_csv(version_output_dir / f"outliers_{current_version}.csv", index=False)

    # Drop "Name" column
    filtered_dataset = filtered_dataset.drop("Name", axis=1)
    outliers = outliers.drop("Name", axis=1)

    # Separate data from labels
    X = filtered_dataset.iloc[:, :-1]
    y = filtered_dataset.iloc[:, -1]
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0, shuffle=True, stratify=y
        )
    except ValueError:
        logging.info("Not enough priority values (other then 0) in the dataset. Abort model generation")
        logging.info("")
        return

    # Add outliers to test sets
    # X_outliers = outliers.iloc[:, :-1]
    # y_outliers = outliers.iloc[:, -1]
    # X_test = pd.concat([X_test, X_outliers], axis=0)
    # y_test = pd.concat([y_test, y_outliers], axis=0)

    # Set 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True)

    # Generate Logistic Regression classifier
    # Optimize the hyperparameters choice with a grid search
    param_grid = {
        "penalty": [None, 'l2', 'l1', 'elasticnet'],
        "solver": ['newton-cg', 'newton-cholesky', 'lbfgs', 'sag', 'saga'],
        "max_iter": [100, 300, 500, 1000]
    }
    existing_model = True
    try:
        with open(version_output_dir / f"logistic_regression_{current_version}.pkl", "rb") as f:
            logistic_regression_clf = load(f)
    except FileNotFoundError:
        existing_model = False
    if not existing_model or recalculate_models:
        logistic_regression_grid = GridSearchCV(LogisticRegression(random_state=42, max_iter=10000), param_grid=param_grid, cv=kf,
                                       scoring='accuracy', verbose=3)
        logistic_regression_grid.fit(X_train, y_train)
        logistic_regression_clf = logistic_regression_grid.best_estimator_
        # Save model
        with open(version_output_dir / f"logistic_regression_{current_version}.pkl", "wb") as f:
            dump(logistic_regression_clf, f, protocol=5)
    logging.info(f"logistic_regression_clf best params: {logistic_regression_clf.get_params()}")
    logging.info(f"logistic_regression_clf coefficients: {logistic_regression_clf.coef_[0]}")
    logging.info(f"logistic_regression_clf intercept_: {logistic_regression_clf.intercept_[0]}")
    logging.info("")

    logging.info("Metrics coefficients:")
    for i, metric in enumerate(filtered_dataset.columns[:-1]):
        logging.info(f"    {metric}: {logistic_regression_clf.coef_[0][i]}")
    logging.info("")

    # Calculate precision and recall for each one of the priorities
    colors = ["cyan", "yellow", "orange", "red"]
    colors_index = 0
    for priority in np.unique(y):
        if priority == "None":
            continue
        # Calculate 10-fold cross validation scores
        # Ref.: https://www.kaggle.com/code/marcinrutecki/gridsearchcv-kfold-cv-the-right-way
        logging.info(f"Calculating scores for {priority} priority...")
        precision = make_scorer(precision_score, pos_label=priority, average="macro")
        precision_score_lr = cross_val_score(logistic_regression_clf, X_train, y_train, cv=kf, scoring=precision)
        lr_precision_score = precision_score_lr.mean()
        lr_precision_stdev = stdev(precision_score_lr)
        logging.info(f'    {priority} priority Logistic Regression Cross Validation Precision scores are: {precision_score_lr}')
        logging.info(f'    {priority} priority Logistic Regression Average Cross Validation Precision score: {lr_precision_score}')
        logging.info(f'    {priority} priority Logistic Regression Cross Validation Precision standard deviation: {lr_precision_stdev}')
        recall = make_scorer(recall_score, pos_label=priority, average="macro")
        recall_score_lr = cross_val_score(logistic_regression_clf, X_train, y_train, cv=kf, scoring=recall)
        lr_recall_score = recall_score_lr.mean()
        lr_recall_stdev = stdev(recall_score_lr)
        logging.info(f'    {priority} priority Logistic Regression Cross Validation Recall scores are: {recall_score_lr}')
        logging.info(f'    {priority} priority Logistic Regression Average Cross Validation Recall score: {lr_recall_score}')
        logging.info(f'    {priority} priority Logistic Regression Cross Validation Recall standard deviation: {lr_recall_stdev}')
        lr_predicted = logistic_regression_clf.predict(X_test)
        lr_predicted_probs = logistic_regression_clf.predict_proba(X_test)
        lr_precision, lr_recall, lr_fscore, lr_support = score(y_test, lr_predicted, average="macro")
        logging.info(f"    {priority} priority Logistic Regression classifier performance:")
        logging.info(f"    {priority} priority precision: {lr_precision}")
        logging.info(f"    {priority} priority recall: {lr_recall}")
        logging.info(f"    {priority} priority fscore: {lr_fscore}")
        logging.info(f"    {priority} priority support: {lr_support}")
        logging.info("")

        # Calculate Logistic Regression AUC
        lr_fpr, lr_tpr, lr_thresholds = roc_curve(y_test, lr_predicted_probs[:, 1], pos_label=priority)
        lr_auc = auc(lr_fpr, lr_tpr)
        logging.info(f"    {priority} Logistic Regression AUC: {lr_auc}")
        logging.info("")

        logging.info(f'{priority} priority scores summary:')
        logging.info(f'    {priority} priority Logistic Regression Average Cross Validation Precision score: {round(lr_precision_score * 100, 1)}')
        logging.info(f'    {priority} priority Logistic Regression Average Cross Validation Recall score: {round(lr_recall_score * 100, 1)}')
        logging.info(f"    {priority} priority precision: {round(lr_precision * 100, 1)}")
        logging.info(f"    {priority} priority recall: {round(lr_recall * 100, 1)}")
        logging.info(f"    {priority} Logistic Regression AUC: {round(lr_auc * 100, 1)}")
        logging.info("")

        # Plot Logistic Regression ROC AUC
        # Ref.: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        label_binarizer = LabelBinarizer().fit(y_train)
        y_onehot_test = label_binarizer.transform(y_test)

        class_of_interest = priority
        class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]

        if plot_images:
            display = RocCurveDisplay.from_predictions(
                y_onehot_test[:, class_id],
                lr_predicted_probs[:, class_id],
                name=f"{class_of_interest} vs the rest",
                color=colors[colors_index],
                plot_chance_level=True,
            )
            other_priorities = " & ".join([p for p in np.unique(y) if p != priority and p != "None"])
            _ = display.ax_.set(
                xlabel="False Positive Rate",
                ylabel="True Positive Rate",
                title=f"{dots_separated_current_version} - One-vs-Rest ROC curves:\n{priority} vs ({other_priorities})",
            )
            display.figure_.savefig(version_output_dir / f"{priority.lower()}_logistic_regression_auc_{current_version}.png")
        colors_index += 1