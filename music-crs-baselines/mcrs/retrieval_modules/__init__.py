from .bm25 import BM25_MODEL
from .bert import BERT_MODEL
from .hybrid_rrf import HYBRID_RRF_MODEL

def load_retrieval_module(
        retrieval_type: str,
        dataset_name: str,
        track_split_types: list[str],
        corpus_types: list[str] = ["track_name", "artist_name", "album_name"],
        cache_dir: str = "./cache",
        **kwargs,
    ):
    if retrieval_type == "bm25":
        return BM25_MODEL(dataset_name, track_split_types, corpus_types, cache_dir)
    elif retrieval_type == "bert":
        return BERT_MODEL(dataset_name, track_split_types, corpus_types, cache_dir)
    elif retrieval_type == "hybrid_rrf":
        return HYBRID_RRF_MODEL(dataset_name, track_split_types, corpus_types, cache_dir, **kwargs)
    else:
        raise ValueError(f"Unsupported retrieval type: {retrieval_type}")
