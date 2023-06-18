
import pytest # noqa
import spacy_setfit  # noqa
import pickle


def test_multi_label(nlp, dataset_multi_label): # noqa
    nlp.add_pipe("text_categorizer", config={
        "pretrained_model_name_or_path": "paraphrase-MiniLM-L3-v2",
        "setfit_trainer_args": {
            "train_dataset": dataset_multi_label,
            "num_iterations": 1
        }
    })

def test_single_label(nlp, dataset_single_label): # noqa
    nlp.add_pipe("text_categorizer", config={
        "pretrained_model_name_or_path": "paraphrase-MiniLM-L3-v2",
        "setfit_trainer_args": {
            "train_dataset": dataset_single_label,
            "num_iterations": 1
        }
    })
    doc = nlp("I really need to get a new sofa.")
    doc.cats

def test_alternative_trainer_args(nlp, dataset_single_label): # noqa
    nlp.add_pipe("text_categorizer", config={
        "pretrained_model_name_or_path": "paraphrase-MiniLM-L3-v2",
        "setfit_trainer_args": {
            "train_dataset": dataset_single_label,
            "eval_dataset": dataset_single_label,
            "num_iterations": 1,
        }
    })

def test_model_without_model_args(nlp, dataset_single_label): # noqa
    nlp.add_pipe("text_categorizer", config={
        "setfit_trainer_args": {
            "train_dataset": dataset_single_label,
            "num_iterations": 1,
        }
    })

def test_model_with_model_args(nlp, dataset_single_label): # noqa
    nlp.add_pipe("text_categorizer", config={
        "setfit_trainer_args": {
            "train_dataset": dataset_single_label,
            "num_iterations": 1,
        },
        "setfit_from_pretrained_args": {
            "force_download": True
        }
    })

def test_save_load_pickle(nlp, dataset_single_label):
    nlp.add_pipe("text_categorizer", config={
        "pretrained_model_name_or_path": "paraphrase-MiniLM-L3-v2",
        "setfit_trainer_args": {
            "train_dataset": dataset_single_label
        }
    })
    doc = nlp("I really need to get a new sofa.")
    doc.cats

    # Save nlp pipeline
    with open("my_cool_model.pkl", "wb") as file:
        pickle.dump(nlp, file)

    # Load nlp pipeline
    with open("my_cool_model.pkl", "rb") as file:
        nlp = pickle.load(file)

    doc = nlp("I really need to get a new sofa.")

def test_without_train_args(nlp):
    with pytest.raises(Exception):
        nlp.add_pipe("text_categorizer", config={
            "pretrained_model_name_or_path": "paraphrase-MiniLM-L3-v2",
        })
        nlp("I really need to get a new sofa.")