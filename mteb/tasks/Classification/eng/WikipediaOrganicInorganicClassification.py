from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikipediaOrganicInorganicClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WikipediaOrganicInorganicClassification",
        description="ChemTEB evaluates the performance of text embedding models on chemical domain data.",
        reference="https://arxiv.org/abs/2412.00532",
        dataset={
            "path": "BASF-AI/WikipediaEasy2SpecialClassification",
            "revision": "96d1d9b37c4693f74c46c83d63a290573f78d511",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-06-01", "2024-11-30"),
        domains=["Chemistry"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@article{kasmaee2024chemteb,
  author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
  journal = {arXiv preprint arXiv:2412.00532},
  title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
  year = {2024},
}
""",
    )
