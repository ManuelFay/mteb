from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class AngryTweetsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AngryTweetsClassification",
        dataset={
            "path": "DDSC/angry-tweets",
            "revision": "20b0e6081892e78179356fada741b7afa381443d",
        },
        description="A sentiment dataset with 3 classes (positiv, negativ, neutral) for Danish tweets",
        reference="https://aclanthology.org/2021.nodalida-main.53/",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["dan-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2021-12-31"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{pauli2021danlp,
  author = {Pauli, Amalie Brogaard and Barrett, Maria and Lacroix, Oph{\'e}lie and Hvingelby, Rasmus},
  booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
  pages = {460--466},
  title = {DaNLP: An open-source toolkit for Danish Natural Language Processing},
  year = {2021},
}
""",
        prompt="Classify Danish tweets by sentiment. (positive, negative, neutral).",
    )

    samples_per_label = 16
