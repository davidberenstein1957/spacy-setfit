
from typing import Union

import pandas as pd
from datasets import ClassLabel, Dataset, Features, Value
from pydantic import BaseModel, root_validator
from sklearn import preprocessing


class SetFitTrainerArgs(BaseModel):
    train_dataset: Union[dict, Dataset]
    eval_dataset: Union[dict, Dataset] = None
    num_iterations: int = 20
    num_epochs: int = 1
    learning_rate: float = 2e-5
    batch_size: float = 16
    seed: int = 42
    column_mapping: dict = None
    use_amp: bool = False
    multi_label: bool = False
    labels: list = None

    class Config:
        arbitrary_types_allowed = True
        fields = {'multi_label': {'exclude': True}, 'labels': {'exclude': True}}

    @root_validator
    def convert_dict_to_dataset(cls, values):
        def _convert_dict_to_dataset(ds_dict):
            df = pd.DataFrame(datasets_dict)
            labels = df.label.unique().tolist()
            df = df.drop_duplicates(subset=["text", "label"])
            df_group = df.groupby("text").agg(list).reset_index()
            if len(df_group) != len(df):
                values["multi_label"] = True
                df = df_group
            return df, labels

        def _create_datasets(df: pd.DataFrame, labels):
            class_label = (
                ClassLabel(names=sorted(labels))
                if labels
                # in case we don't have any labels, ClassLabel fails with Dataset.from_dict({"labels": []})
                else Value(dtype='int64', id=None)
            )

            feature_dict = {
                "text": Value("string"),
                "label": [class_label] if values["multi_label"] else class_label,
            }

            print(df.to_dict(orient="list"))
            ds = Dataset.from_dict(df.to_dict(orient="list"), features=Features(feature_dict))
            ds = ds.shuffle(seed=values["seed"])

            return ds

        def _add_data_to_dict(datasets_dict, data, train_or_test):
            for label, texts in data.items():
                for text in texts:
                    datasets_dict["text"].append(text)
                    datasets_dict["label"].append(label)
                    datasets_dict["split"].append(train_or_test)
            return datasets_dict

        if isinstance(values["train_dataset"], dict):
            options = ["train_dataset"]
            datasets_dict = {"text": [], "label": [], "split": []}
            datasets_dict = _add_data_to_dict(datasets_dict, values["train_dataset"], "train_dataset")
            if isinstance(values["eval_dataset"], dict):
                options.append("eval_dataset")
                datasets_dict = _add_data_to_dict(datasets_dict, values["eval_dataset"], "eval_dataset")
            elif isinstance(values["eval_dataset"], Dataset):
                raise ValueError("train_dataset and eval_dataset must be of the same type")

            df, labels = _convert_dict_to_dataset(datasets_dict)
            values["labels"] = labels

            if values["multi_label"]:
                le = preprocessing.MultiLabelBinarizer()
            else:
                le = preprocessing.LabelEncoder()
            df["label"] = le.fit_transform(df["label"]).tolist()

            for train_or_test in options:
                if values["multi_label"]:
                    df_filtered = df.copy(deep=True)
                    df_filtered["split"] = df_filtered["split"].apply(lambda x: True if train_or_test in x else False)
                    df_filtered = df_filtered[df_filtered["split"] == True] # noqa
                    df_filtered["label"] = df_filtered["label"].apply(lambda x: x)
                else:
                    df_filtered = df[df["split"] == train_or_test]

                if not df_filtered.empty:
                    df_filtered = df_filtered.drop(columns=["split"])
                    values[train_or_test] = _create_datasets(df_filtered, [0,1])
            return values
        else:
            return values

