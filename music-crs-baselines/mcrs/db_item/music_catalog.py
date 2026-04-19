from datasets import load_dataset, concatenate_datasets

class MusicCatalogDB:
    def __init__(self,
            dataset_name: str,
            split_types: list[str],
            corpus_types: list[str],
        ):
        metadata_dataset = load_dataset(dataset_name)
        metadata_concat_dataset = concatenate_datasets([metadata_dataset[split_type] for split_type in split_types])
        self.corpus_types = corpus_types
        self.metadata_dict = {item["track_id"]: item for item in metadata_concat_dataset}

    def _format_metadata_value(self, value):
        if isinstance(value, list):
            return ", ".join(str(item) for item in value)
        return str(value)

    def id_to_metadata(self, track_id: str, use_semantic_id: bool = False, include_track_id: bool = False, extra_fields: list[str] | None = None):
        metadata = self.metadata_dict[track_id]
        fields = list(dict.fromkeys(self.corpus_types + (extra_fields or [])))
        entity_parts = []
        if include_track_id:
            entity_parts.append(f"track_id: {metadata['track_id']}")
        for corpus_type in fields:
            if corpus_type not in metadata:
                continue
            entity_parts.append(f"{corpus_type}: {self._format_metadata_value(metadata[corpus_type])}")
        entity_str = ", ".join(entity_parts)
        return entity_str

    def ids_to_metadata_block(self, track_ids: list[str], extra_fields: list[str] | None = None, include_track_id: bool = False) -> str:
        lines = []
        for idx, track_id in enumerate(track_ids, start=1):
            lines.append(
                f"{idx}. {self.id_to_metadata(track_id, include_track_id=include_track_id, extra_fields=extra_fields)}"
            )
        return "\n".join(lines)
