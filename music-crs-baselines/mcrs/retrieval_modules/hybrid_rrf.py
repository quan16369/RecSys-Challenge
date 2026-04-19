import re
from collections import defaultdict

from .bm25 import BM25_MODEL
from .cf import CF_EMBEDDING_MODEL


class HYBRID_RRF_MODEL:
    def __init__(
        self,
        dataset_name: str,
        split_types: list[str],
        corpus_types: list[str],
        cache_dir: str = "./cache",
        device: str = "cuda",
        track_embedding_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        user_embedding_db_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
        user_embedding_split_types: list[str] | None = None,
        rrf_k: int = 60,
        bm25_candidate_k: int = 200,
        cf_candidate_k: int = 200,
    ) -> None:
        self.bm25_model = BM25_MODEL(dataset_name, split_types, corpus_types, cache_dir)
        self.cf_model = CF_EMBEDDING_MODEL(
            track_dataset_name=track_embedding_db_name,
            user_dataset_name=user_embedding_db_name,
            track_split_types=split_types,
            user_split_types=user_embedding_split_types,
            device=device,
        )
        self.metadata_dict = self.bm25_model.metadata_dict
        self.rrf_k = rrf_k
        self.bm25_candidate_k = bm25_candidate_k
        self.cf_candidate_k = cf_candidate_k
        self.global_popularity_ids = [
            track_id
            for track_id, _ in sorted(
                self.metadata_dict.items(),
                key=lambda item: float(item[1].get("popularity") or 0.0),
                reverse=True,
            )
        ]

    def _clean_text(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return ", ".join(str(item) for item in value if item)
        return str(value)

    def _metadata_text(self, track_id: str) -> str:
        metadata = self.metadata_dict.get(track_id, {})
        values = [
            self._clean_text(metadata.get("track_name")),
            self._clean_text(metadata.get("artist_name")),
            self._clean_text(metadata.get("album_name")),
            self._clean_text(metadata.get("tag_list")),
            self._clean_text(metadata.get("release_date")),
        ]
        return " ".join(part for part in values if part)

    def _profile_text(self, context: dict) -> str:
        user_profile = context.get("user_profile") or {}
        values = [
            user_profile.get("preferred_musical_culture"),
            user_profile.get("preferred_language"),
            user_profile.get("country_name"),
            user_profile.get("age_group"),
            user_profile.get("gender"),
        ]
        return " ".join(self._clean_text(value) for value in values if value)

    def _goal_text(self, context: dict) -> str:
        conversation_goal = context.get("conversation_goal") or {}
        values = [
            conversation_goal.get("listener_goal"),
            conversation_goal.get("category"),
            conversation_goal.get("specificity"),
        ]
        return " ".join(self._clean_text(value) for value in values if value)

    def _seed_text(self, track_ids: list[str]) -> str:
        snippets = [self._metadata_text(track_id) for track_id in track_ids[:3]]
        return " ".join(snippets)

    def _focused_query(self, full_context_query: str, context: dict) -> str:
        current_query = context.get("user_query", "")
        parts = [
            current_query,
            self._goal_text(context),
            self._profile_text(context),
            self._seed_text(context.get("positive_track_ids", [])),
        ]
        if not any(parts):
            return full_context_query
        return "\n".join(part for part in parts if part)

    def _query_terms(self, query: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]{3,}", query.lower())
            if token not in {"with", "from", "that", "this", "your", "have", "like"}
        }

    def _year_bonus(self, query: str, track_id: str) -> float:
        years = re.findall(r"\b(19\d{2}|20\d{2})\b", query)
        decades = re.findall(r"\b(19\d0|20\d0)s\b", query.lower())
        metadata = self.metadata_dict.get(track_id, {})
        release_date = str(metadata.get("release_date") or "")
        if not release_date:
            return 0.0
        bonus = 0.0
        for year in years:
            if release_date.startswith(year):
                bonus += 0.03
        for decade in decades:
            if release_date.startswith(decade[:3]):
                bonus += 0.02
        return bonus

    def _keyword_bonus(self, query_terms: set[str], track_id: str) -> float:
        if not query_terms:
            return 0.0
        track_terms = self._query_terms(self._metadata_text(track_id))
        overlap = len(query_terms.intersection(track_terms))
        return min(overlap, 6) * 0.005

    def _fallback_fill(self, ranked_track_ids: list[str], exclude_track_ids: set[str], topk: int) -> list[str]:
        seen = set(ranked_track_ids)
        for track_id in self.global_popularity_ids:
            if track_id in seen or track_id in exclude_track_ids:
                continue
            ranked_track_ids.append(track_id)
            seen.add(track_id)
            if len(ranked_track_ids) >= topk:
                break
        return ranked_track_ids

    def _fuse_rankings(
        self,
        rankings: dict[str, list[str]],
        context: dict,
        topk: int,
    ) -> list[str]:
        query = self._focused_query("", context)
        query_terms = self._query_terms(query)
        exclude_track_ids = set(context.get("history_track_ids", []))
        scores = defaultdict(float)
        weights = {
            "bm25_focused": 1.35,
            "bm25_full_context": 1.0,
            "user_cf": 0.75,
            "seed_cf": 1.15,
        }

        for source_name, ranking in rankings.items():
            weight = weights.get(source_name, 1.0)
            for rank, track_id in enumerate(ranking, start=1):
                if track_id in exclude_track_ids:
                    continue
                scores[track_id] += weight / (self.rrf_k + rank)

        for track_id in list(scores):
            popularity = float(self.metadata_dict.get(track_id, {}).get("popularity") or 0.0)
            scores[track_id] += popularity / 10000.0
            scores[track_id] += self._keyword_bonus(query_terms, track_id)
            scores[track_id] += self._year_bonus(query, track_id)

        ranked_track_ids = [
            track_id
            for track_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ]
        ranked_track_ids = ranked_track_ids[:topk]
        return self._fallback_fill(ranked_track_ids, exclude_track_ids, topk)

    def batch_text_to_item_retrieval_with_context(
        self,
        full_context_queries: list[str],
        contexts: list[dict],
        topk: int,
    ) -> list[list[str]]:
        focused_queries = [self._focused_query(full_context_query, context) for full_context_query, context in zip(full_context_queries, contexts)]

        bm25_focused = self.bm25_model.batch_text_to_item_retrieval(focused_queries, topk=self.bm25_candidate_k)
        bm25_full_context = self.bm25_model.batch_text_to_item_retrieval(full_context_queries, topk=self.bm25_candidate_k)

        exclude_track_ids_batch = [set(context.get("history_track_ids", [])) for context in contexts]
        user_cf = self.cf_model.batch_user_to_item_retrieval(
            [context.get("user_id") for context in contexts],
            topk=self.cf_candidate_k,
            exclude_track_ids_batch=exclude_track_ids_batch,
        )
        seed_track_ids_batch = []
        for context in contexts:
            seed_track_ids = context.get("positive_track_ids") or context.get("recent_track_ids") or []
            seed_track_ids_batch.append(seed_track_ids)
        seed_cf = self.cf_model.batch_seed_tracks_to_item_retrieval(
            seed_track_ids_batch,
            topk=self.cf_candidate_k,
            exclude_track_ids_batch=exclude_track_ids_batch,
        )

        batch_results = []
        for idx, context in enumerate(contexts):
            rankings = {
                "bm25_focused": bm25_focused[idx],
                "bm25_full_context": bm25_full_context[idx],
                "user_cf": user_cf[idx],
                "seed_cf": seed_cf[idx],
            }
            batch_results.append(self._fuse_rankings(rankings, context, topk))
        return batch_results

    def text_to_item_retrieval(self, query: str, topk: int) -> list[str]:
        return self.batch_text_to_item_retrieval_with_context([query], [{"user_query": query}], topk)[0]
