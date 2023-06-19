import logging
import types

from setfit import SetFitModel, SetFitTrainer
from spacy import util
from spacy.language import Language
from spacy.tokens import Doc

from spacy_setfit.schemas import SetFitTrainerArgs

__LOGGER__ = logging.getLogger(__name__)


class SpacySetFit:
    def __init__(self, nlp: Language, model: SetFitModel, labels=None):
        self.nlp = nlp
        self.model = model
        self.multi_label = self._check_multi_label(model)
        self.labels = labels
        if self.labels:
            self.id2label = {i: label for i, label in enumerate(self.labels)}
        else:
            self.id2label = None

    @staticmethod
    def _check_multi_label(model: SetFitModel):
        if model.multi_target_strategy:
            return True
        else:
            return False

    @staticmethod
    def _from_pretrained(
        pretrained_model_name_or_path: str, setfit_from_pretrained_args: dict = None
    ):
        if setfit_from_pretrained_args is None:
            setfit_from_pretrained_args = {}
        model = SetFitModel.from_pretrained(
            pretrained_model_name_or_path, **setfit_from_pretrained_args
        )
        return model

    @classmethod
    def from_pretrained(
        cls,
        nlp: Language,
        pretrained_model_name_or_path: str,
        setfit_from_pretrained_args: dict = None,
    ):
        model = cls._from_pretrained(
            pretrained_model_name_or_path, setfit_from_pretrained_args
        )
        return cls(nlp, model)

    @classmethod
    def from_trained(
        cls,
        nlp: Language,
        pretrained_model_name_or_path: str,
        setfit_trainer_args: SetFitTrainerArgs,
        setfit_from_pretrained_args: dict = None,
    ):
        setfit_from_pretrained_args[
            "multi_target_strategy"
        ] = setfit_from_pretrained_args.get("multi_target_strategy")
        if setfit_trainer_args.multi_label:
            setfit_from_pretrained_args["multi_target_strategy"] = "one-vs-rest"

        model = cls._from_pretrained(
            pretrained_model_name_or_path, setfit_from_pretrained_args
        )
        trainer = SetFitTrainer(model=model, **setfit_trainer_args.dict())
        trainer.train()
        if setfit_trainer_args.eval_dataset:
            evaluation = trainer.evaluate()
            __LOGGER__.info(f"evaluation: {evaluation}")

        return cls(nlp, model, labels=setfit_trainer_args.labels)

    def _assign_labels(self, doc, prediction: list):
        if self.id2label:
            doc.cats = {
                self.id2label[idx]: float(score) for idx, score in enumerate(prediction)
            }
        else:
            doc.cats = {idx: float(score) for idx, score in enumerate(prediction)}
        return doc

    def __call__(self, doc: Doc):
        """
        It takes a doc, gets the embeddings from the doc, reshapes the embeddings, gets the prediction from the embeddings,
        and then sets the prediction results for the doc

        :param doc: Doc
        :type doc: Doc
        :return: The doc object with the predicted categories and the predicted categories for each sentence.
        """
        if isinstance(doc, str):
            doc = self.nlp(doc)
        prediction = self.model.predict_proba([doc.text])
        doc = self._assign_labels(doc, prediction[0])

        return doc

    def pipe(self, stream, batch_size=128, include_sent=None):
        """
        predict the class for a spacy Doc stream

        Args:
            stream (Doc): a spacy doc

        Returns:
            Doc: spacy doc with ._.cats key-class proba-value dict
        """
        if isinstance(stream, str):
            stream = [stream]

        if not isinstance(stream, types.GeneratorType):
            stream = self.nlp.pipe(stream, batch_size=batch_size)

        for docs in util.minibatch(stream, size=batch_size):
            pred_results = self.model.predict_proba([doc.text for doc in docs])

            for doc, prediction in zip(docs, pred_results):
                yield self._assign_labels(doc, prediction)
