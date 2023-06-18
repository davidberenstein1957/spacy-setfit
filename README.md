# spacy-setfit

This repository contains an easy and intuitive approach to using [SetFit](https://github.com/huggingface/setfit) in combination with [spaCy](https://github.com/explosion/spaCy).

## Installation

Before using spaCy with SetFit, make sure you have the necessary packages installed. You can install them using pip:

```
pip install spacy spacy-setfit
```

Additionally, you will need to download a spaCy model, for example:

```
python -m spacy download en_core_web_sm
```

## Getting Started

To use spaCy with SetFit use the following code:

```python
import spacy
import spacy_setfit

# Create some example data
train_dataset = {
    "inlier": ["This text is about chairs.",
               "Couches, benches and televisions.",
               "I really need to get a new sofa."],
    "outlier": ["Text about kitchen equipment",
                "This text is about politics",
                "Comments about AI and stuff."]
}

# Load the spaCy language model:
nlp = spacy.load("en_core_web_sm")

# Add the "text_categorizer" pipeline component to the spaCy model, and configure it with SetFit parameters:
nlp.add_pipe("text_categorizer", config={
    "pretrained_model_name_or_path": "paraphrase-MiniLM-L3-v2",
    "setfit_trainer_args": {
        "train_dataset": train_dataset
    }
})
doc = nlp("I really need to get a new sofa.")
doc.cats
# {'inlier': 0.902350975129, 'outlier': 0.097649024871}
```

The code above processes the input text with the spaCy model, and the `doc.cats` attribute returns the predicted categories and their associated probabilities.

That's it! You have now successfully integrated spaCy with SetFit for text categorization tasks. You can further customize and train the model using additional data or adjust the SetFit parameters as needed.

Feel free to explore more features and documentation of spaCy and SetFit to enhance your text classification projects.

## setfit_trainer_args

The `setfit_trainer_args` are a simplified version of the official args from the SetFit library.

### Arguments

- `train_dataset` (Union[dict, Dataset]): The training dataset to be used by the SetFitTrainer. It can be either a dictionary or a Dataset object.

- `eval_dataset` (Union[dict, Dataset], optional): The evaluation dataset to be used by the SetFitTrainer. It can be either a dictionary or a Dataset object. Defaults to `None`.

- `num_iterations` (int, optional): The number of iterations to train the model. Defaults to `20`.

- `num_epochs` (int, optional): The number of epochs to train the model. Defaults to `1`.

- `learning_rate` (float, optional): The learning rate for the optimizer. Defaults to `2e-5`.

- `batch_size` (float, optional): The batch size for training. Defaults to `16`.

- `seed` (int, optional): The random seed for reproducibility. Defaults to `42`.

- `column_mapping` (dict, optional): A mapping dictionary that specifies how to map input columns to model inputs. Defaults to `None`.

- `use_amp` (bool, optional): Whether to use Automatic Mixed Precision (AMP) for training. Defaults to `False`.

### Usage

To use the `setfit_trainer_args`, you can create an instance of the class and set the desired values for the arguments. Here's an example:

```python
args = {
    "train_dataset": train_data,
    "eval_dataset": eval_data,
    "num_iterations": 20,
    "num_epochs": 1,
    "learning_rate": 2e-5,
    "batch_size": 16,
    "seed": 42,
    "column_mapping": column_map,
    "use_amp": False
}
```

## Saving and Loading models

## Saving and Loading with Pickle

You can use the `pickle` module in Python to save and load instances of the pre-trained pipeline. `pickle` allows you to serialize Python objects, including custom classes, into a binary format that can be saved to a file and loaded back into memory later. Here's an example of how to save and load using `pickle`:

```python
import pickle

nlp = ...

# Save nlp pipeline
with open("my_cool_model.pkl", "wb") as file:
    pickle.dump(nlp, file)

# Load nlp pipeline
with open("my_cool_model.pkl", "rb") as file:
    nlp = pickle.load(file)

doc = nlp("I really need to get a new sofa.")
doc.cats
# {'inlier': 0.902350975129, 'outlier': 0.097649024871}
```
