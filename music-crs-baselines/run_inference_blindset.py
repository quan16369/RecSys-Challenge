"""
Batch inference script for Music CRS.
"""

import os
import json
import torch
import argparse
from mcrs import load_crs_baseline
from mcrs.inference_context import build_blind_context
from datasets import load_dataset
from tqdm import tqdm
from omegaconf import OmegaConf

def resolve_torch_dtype(config):
    dtype_name = getattr(config, "dtype", "bfloat16")
    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch dtype in config: {dtype_name}")
    return getattr(torch, dtype_name)

def main(args):
    """
    Run batch inference on TalkPlayData-2 test dataset.

    Args:
        args: Namespace object containing:
            - tid (str): Task/configuration identifier
            - batch_size (int): Batch size for inference
            - save_path (str): Output directory (currently unused)

    Returns:
        None. Results are saved to exp/inference/{tid}.json

    Processing:
        - Loads model configuration from config/{tid}.yaml
        - Processes all sessions × 8 turns in batches
        - Tracks progress with tqdm progress bar
        - Saves comprehensive results for evaluation
    """
    if args.refresh_cache:
        print("Refreshing cache directory...")
        os.system("rm -rf cache")
    config = OmegaConf.load(f"config/{args.tid}.yaml")
    music_crs = load_crs_baseline(
        lm_type=config.lm_type,
        retrieval_type=config.retrieval_type,
        item_db_name=config.item_db_name,
        user_db_name=config.user_db_name,
        track_split_types=config.track_split_types,
        user_split_types=config.user_split_types,
        corpus_types=config.corpus_types,
        cache_dir=config.cache_dir,
        device=config.device,
        attn_implementation=config.attn_implementation,
        dtype=resolve_torch_dtype(config),
        track_embedding_db_name=getattr(config, "track_embedding_db_name", "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"),
        user_embedding_db_name=getattr(config, "user_embedding_db_name", "talkpl-ai/TalkPlayData-Challenge-User-Embeddings"),
        user_embedding_split_types=getattr(config, "user_embedding_split_types", ["train", "test_warm", "test_cold"]),
        rrf_k=getattr(config, "rrf_k", 60),
        bm25_candidate_k=getattr(config, "bm25_candidate_k", 200),
        cf_candidate_k=getattr(config, "cf_candidate_k", 200),
        max_input_tokens=getattr(config, "max_input_tokens", 1536),
        max_new_tokens=getattr(config, "max_new_tokens", 48),
    )
    db = load_dataset(config.test_dataset_name, split="test")
    # Prepare all batch data at once
    batch_data, metadata = [], []
    for item in db:
        user_id = item['user_id']
        session_id = item['session_id']
        chat_history, user_query, retrieval_context, turn_number = build_blind_context(item, music_crs)
        batch_data.append({
            'user_query': user_query,
            'user_id': user_id,
            'session_memory': chat_history,
            'retrieval_context': retrieval_context,
        })
        metadata.append({
            'session_id': session_id,
            'user_id': user_id,
            'turn_number': turn_number
        })
    inference_results = []
    for i in tqdm(range(0, len(batch_data), args.batch_size), desc="Batch inference"):
        batch = batch_data[i:i+args.batch_size]
        batch_metadata = metadata[i:i+args.batch_size]
        results = music_crs.batch_chat(batch)
        for j, result in enumerate(results):
            inference_results.append({
                "session_id": batch_metadata[j]['session_id'],
                "user_id": batch_metadata[j]['user_id'],
                "turn_number": batch_metadata[j]['turn_number'],
                "predicted_track_ids": result['retrieval_items'],
                "predicted_response": result["response"]
            })
    os.makedirs(f"exp/inference/{args.eval_dataset}", exist_ok=True)
    with open(f"exp/inference/{args.eval_dataset}/{args.tid}.json", "w", encoding="utf-8") as f:
        json.dump(inference_results, f, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batch inference on TalkPlayData-2 test dataset for Music CRS evaluation."
    )
    parser.add_argument(
        "--tid",
        type=str,
        default="llama1b_bm25_blindset_A",
        help="Task identifier matching a config file (e.g., 'llama1b_bm25' loads config/llama1b_bm25.yaml)"
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="blindset_A",
        help="Evaluation dataset name"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of queries to process in parallel. Reduce if encountering GPU memory issues."
    )
    parser.add_argument(
        "--refresh_cache",
        action="store_true",
        help="Rebuild retrieval caches instead of reusing the existing cache directory."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./exp/inference",
        help="Base directory for saving results (currently not used, results saved to exp/inference/)"
    )
    args = parser.parse_args()
    main(args)
