import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset


class CF_EMBEDDING_MODEL:
    def __init__(
        self,
        track_dataset_name: str,
        user_dataset_name: str,
        track_split_types: list[str],
        user_split_types: list[str] | None = None,
        embedding_field: str = "cf-bpr",
        device: str | None = None,
    ) -> None:
        self.track_dataset_name = track_dataset_name
        self.user_dataset_name = user_dataset_name
        self.track_split_types = track_split_types
        self.user_split_types = user_split_types or ["train", "test_warm", "test_cold"]
        self.embedding_field = embedding_field
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.track_ids, self.track_embeddings = self._load_track_embeddings()
        self.track_id_to_index = {track_id: idx for idx, track_id in enumerate(self.track_ids)}
        self.user_embeddings = self._load_user_embeddings()

    def _select_splits(self, dataset_dict, split_types: list[str]):
        available_splits = [split for split in split_types if split in dataset_dict]
        if available_splits:
            return available_splits
        return list(dataset_dict.keys())

    def _load_track_embeddings(self) -> tuple[list[str], torch.Tensor]:
        dataset = load_dataset(self.track_dataset_name)
        split_types = self._select_splits(dataset, self.track_split_types)
        rows = concatenate_datasets([dataset[split_type] for split_type in split_types])

        track_ids = []
        embeddings = []
        for row in rows:
            embedding = row.get(self.embedding_field)
            if not embedding:
                continue
            track_ids.append(row["track_id"])
            embeddings.append(embedding)

        track_embeddings = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        track_embeddings = F.normalize(track_embeddings, p=2, dim=1)
        return track_ids, track_embeddings

    def _load_user_embeddings(self) -> dict[str, torch.Tensor]:
        dataset = load_dataset(self.user_dataset_name)
        split_types = self._select_splits(dataset, self.user_split_types)
        rows = concatenate_datasets([dataset[split_type] for split_type in split_types])

        user_embeddings = {}
        for row in rows:
            embedding = row.get(self.embedding_field)
            if not embedding:
                continue
            user_embeddings[row["user_id"]] = F.normalize(
                torch.tensor(embedding, dtype=torch.float32, device=self.device),
                p=2,
                dim=0,
            )
        return user_embeddings

    def _topk_from_scores(
        self,
        scores: torch.Tensor,
        topk: int,
        exclude_track_ids: set[str] | None = None,
    ) -> list[str]:
        exclude_track_ids = exclude_track_ids or set()
        ordered_indices = torch.argsort(scores, descending=True)
        results = []
        for index in ordered_indices.tolist():
            track_id = self.track_ids[index]
            if track_id in exclude_track_ids:
                continue
            results.append(track_id)
            if len(results) >= topk:
                break
        return results

    def batch_user_to_item_retrieval(
        self,
        user_ids: list[str | None],
        topk: int,
        exclude_track_ids_batch: list[set[str]] | None = None,
    ) -> list[list[str]]:
        exclude_track_ids_batch = exclude_track_ids_batch or [set() for _ in user_ids]
        valid = [(idx, self.user_embeddings.get(user_id)) for idx, user_id in enumerate(user_ids)]
        valid = [(idx, embedding) for idx, embedding in valid if embedding is not None]
        results = [[] for _ in user_ids]
        if not valid:
            return results

        batch_embeddings = torch.stack([embedding for _, embedding in valid], dim=1)
        batch_scores = torch.matmul(self.track_embeddings, batch_embeddings)

        for col_idx, (row_idx, _) in enumerate(valid):
            results[row_idx] = self._topk_from_scores(
                batch_scores[:, col_idx],
                topk=topk,
                exclude_track_ids=exclude_track_ids_batch[row_idx],
            )
        return results

    def batch_seed_tracks_to_item_retrieval(
        self,
        seed_track_ids_batch: list[list[str]],
        topk: int,
        exclude_track_ids_batch: list[set[str]] | None = None,
    ) -> list[list[str]]:
        exclude_track_ids_batch = exclude_track_ids_batch or [set() for _ in seed_track_ids_batch]
        seed_embeddings = []
        valid_indices = []

        for idx, seed_track_ids in enumerate(seed_track_ids_batch):
            valid_seed_ids = [track_id for track_id in seed_track_ids if track_id in self.track_id_to_index]
            if not valid_seed_ids:
                continue
            indices = [self.track_id_to_index[track_id] for track_id in valid_seed_ids]
            seed_embedding = self.track_embeddings[indices].mean(dim=0)
            seed_embedding = F.normalize(seed_embedding, p=2, dim=0)
            seed_embeddings.append(seed_embedding)
            valid_indices.append(idx)

        results = [[] for _ in seed_track_ids_batch]
        if not seed_embeddings:
            return results

        batch_embeddings = torch.stack(seed_embeddings, dim=1)
        batch_scores = torch.matmul(self.track_embeddings, batch_embeddings)

        for col_idx, row_idx in enumerate(valid_indices):
            results[row_idx] = self._topk_from_scores(
                batch_scores[:, col_idx],
                topk=topk,
                exclude_track_ids=exclude_track_ids_batch[row_idx],
            )
        return results
