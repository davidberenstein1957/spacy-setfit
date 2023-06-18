import pytest


@pytest.fixture
def nlp():
    import spacy
    nlp = spacy.blank("en")
    return nlp

@pytest.fixture
def dataset_multi_label():
    return {
        "inlier": ["This text is about chairs.",
                   "Couches, benches and televisions.",
                   "My couch is old and I need a new one.",
                   "I really need to get a new sofa."],
        "outlier": ["Text about kitchen equipment",
                    "This text is about politics",
                    "Comments about AI and stuff.",
                    "This text is about chairs.",
                    "I really need to get a new sofa."]
    }

@pytest.fixture
def dataset_single_label():
    return {
        "inlier": ["This text is about chairs.",
                   "Couches, benches and televisions.",
                   "I really need to get a new sofa."],
        "outlier": ["Text about kitchen equipment",
                    "This text is about politics",
                    "Comments about AI and stuff."]
    }

