from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class LitSearchRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LitSearchRetrieval",
        description="""
        The dataset contains the query set and retrieval corpus for the paper LitSearch: A Retrieval Benchmark for
        Scientific Literature Search. It introduces LitSearch, a retrieval benchmark comprising 597 realistic literature
        search queries about recent ML and NLP papers. LitSearch is constructed using a combination of (1) questions
        generated by GPT-4 based on paragraphs containing inline citations from research papers and (2) questions about
        recently published papers, manually written by their authors. All LitSearch questions were manually examined or
        edited by experts to ensure high quality.
        """,
        reference="https://github.com/princeton-nlp/LitSearch",
        dataset={
            "path": "princeton-nlp/LitSearch",
            "revision": "9573fb284a1026c998df47024b888a163f0f0e25",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-07-10", "2024-05-11"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="LM-generated",  # generated by GPT-4
        dialect=[],
        sample_creation="found",  # queries LLM generated, corpus samples are found (extracted from S2ORC)
        bibtex_citation=r"""
@article{ajith2024litsearch,
  author = {Ajith, Anirudh and Xia, Mengzhou and Chevalier, Alexis and Goyal, Tanya and Chen, Danqi and Gao, Tianyu},
  title = {LitSearch: A Retrieval Benchmark for Scientific Literature Search},
  year = {2024},
}
""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        dataset_path = self.metadata_dict["dataset"]["path"]

        query_ds = datasets.load_dataset(dataset_path, "query")

        self.queries["test"] = dict(
            zip(
                [f"q{x + 1}" for x in range(len(query_ds["full"]))],
                query_ds["full"]["query"],
            )
        )

        corpus_ds = datasets.load_dataset(dataset_path, "corpus_clean")

        self.corpus["test"] = {
            f"d{c_id}": {"title": title, "text": txt}
            for c_id, title, txt in zip(
                corpus_ds["full"]["corpusid"],
                corpus_ds["full"]["title"],
                corpus_ds["full"]["abstract"],
            )
        }

        self.relevant_docs["test"] = {
            f"q{e + 1}": dict(zip([f"d{i}" for i in ids], range(1, len(ids) + 1)))
            for e, ids in enumerate(query_ds["full"]["corpusids"])
        }

        self.data_loaded = True
