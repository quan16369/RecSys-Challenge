from typing import Any


def _assessment_map(goal_progress_assessments: list[dict[str, Any]] | None) -> dict[int, str]:
    assessments = {}
    for item in goal_progress_assessments or []:
        turn_number = item.get("turn_number")
        assessment = item.get("goal_progress_assessment")
        if turn_number is None or assessment is None:
            continue
        assessments[int(turn_number)] = assessment
    return assessments


def _music_turn_to_history_message(turn_data: dict[str, Any], music_crs) -> dict[str, str]:
    metadata = music_crs.item_db.id_to_metadata(
        turn_data["content"],
        extra_fields=["tag_list", "release_date", "popularity"],
    )
    content_parts = [f"recommended_track: {metadata}"]
    thought = turn_data.get("thought")
    if thought:
        content_parts.append(f"recommendation_rationale: {thought}")
    return {
        "role": "assistant",
        "content": "\n".join(content_parts),
    }


def build_turn_context(item: dict[str, Any], music_crs, target_turn_number: int) -> tuple[list[dict[str, str]], str, dict[str, Any]]:
    conversations = item["conversations"]
    chat_history = []
    history_track_ids = []
    history_music_by_turn = []

    for turn_data in conversations:
        turn_number = int(turn_data["turn_number"])
        if turn_number >= target_turn_number:
            continue
        if turn_data["role"] == "music":
            history_track_ids.append(turn_data["content"])
            history_music_by_turn.append((turn_number, turn_data["content"]))
            chat_history.append(_music_turn_to_history_message(turn_data, music_crs))
            continue
        chat_history.append({
            "role": turn_data["role"],
            "content": turn_data["content"],
        })

    current_turn = next(turn for turn in conversations if int(turn["turn_number"]) == target_turn_number and turn["role"] == "user")
    user_query = current_turn["content"]

    assessment_by_turn = _assessment_map(item.get("goal_progress_assessments"))
    positive_track_ids = []
    negative_track_ids = []
    for turn_number, track_id in history_music_by_turn:
        assessment = assessment_by_turn.get(turn_number + 1)
        if assessment == "MOVES_TOWARD_GOAL":
            positive_track_ids.append(track_id)
        elif assessment == "DOES_NOT_MOVE_TOWARD_GOAL":
            negative_track_ids.append(track_id)

    retrieval_context = {
        "session_id": item.get("session_id"),
        "turn_number": target_turn_number,
        "history_track_ids": history_track_ids,
        "recent_track_ids": history_track_ids[-3:],
        "positive_track_ids": positive_track_ids,
        "negative_track_ids": negative_track_ids,
        "user_profile": item.get("user_profile", {}),
        "conversation_goal": item.get("conversation_goal", {}),
    }
    return chat_history, user_query, retrieval_context


def build_blind_context(item: dict[str, Any], music_crs) -> tuple[list[dict[str, str]], str, dict[str, Any], int]:
    conversations = item["conversations"]
    current_turn = conversations[-1]
    target_turn_number = int(current_turn["turn_number"])

    history_item = {
        **item,
        "conversations": conversations,
    }
    chat_history, user_query, retrieval_context = build_turn_context(history_item, music_crs, target_turn_number)
    return chat_history, user_query, retrieval_context, target_turn_number
