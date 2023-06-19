import logging

from rich.logging import RichHandler
from spacy.language import Language

from spacy_setfit.models import SpacySetFit
from spacy_setfit.schemas import SetFitTrainerArgs

__all__ = ["SpacySetFit", "SetFitTrainerArgs"]

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)


@Language.factory(
    "text_categorizer",
    default_config={
        "pretrained_model_name_or_path": "all-MiniLM-L6-v2",
        "setfit_from_pretrained_args": None,
        "setfit_trainer_args": None,
    },
)
def create_setfit_model(
    nlp: Language,
    name: str,
    pretrained_model_name_or_path: str = "all-MiniLM-L6-v2",
    setfit_from_pretrained_args: dict = None,
    setfit_trainer_args: dict = None,
):
    if setfit_from_pretrained_args is None:
        setfit_from_pretrained_args = {}
    if setfit_trainer_args is None:
        setfit_trainer_args = {}
    else:
        setfit_trainer_args = SetFitTrainerArgs(**setfit_trainer_args)

    if setfit_trainer_args != {}:
        return SpacySetFit.from_trained(
            nlp=nlp,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            setfit_from_pretrained_args=setfit_from_pretrained_args,
            setfit_trainer_args=setfit_trainer_args,
        )
    else:
        return SpacySetFit.from_pretrained(
            nlp=nlp,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            setfit_from_pretrained_args=setfit_from_pretrained_args,
        )
