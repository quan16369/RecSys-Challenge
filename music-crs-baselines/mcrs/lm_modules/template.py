import re


class TEMPLATE_MODEL:
    def __init__(self, model_name="template", device="cpu", attn_implementation="eager", dtype=None):
        self.model_name = model_name
        self.device = device
        self.attn_implementation = attn_implementation
        self.dtype = dtype

    def _extract_tracks(self, recommend_item: str) -> list[dict[str, str]]:
        tracks = []
        for line in (recommend_item or "").splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^\d+\.\s*", "", line)
            parts = {}
            for field in ["track_name", "artist_name", "album_name", "tag_list", "release_date", "popularity"]:
                match = re.search(rf"{field}:\s*(.*?)(?=,\s+[a-z_]+:|$)", line)
                if match:
                    parts[field] = match.group(1).strip()
            if parts:
                tracks.append(parts)
        return tracks

    def _render_response(self, recommend_item: str) -> str:
        tracks = self._extract_tracks(recommend_item)
        if not tracks:
            return "I found a few possible matches. If you want, I can narrow them down by mood, era, or genre."

        lead = tracks[0]
        title = lead.get("track_name", "this track")
        artist = lead.get("artist_name", "the artist")
        tags = lead.get("tag_list", "")
        release_date = lead.get("release_date", "")

        details = []
        if tags:
            details.append(tags.split(",")[0].strip())
        if release_date:
            details.append(release_date[:4])
        detail_str = ", ".join(detail for detail in details if detail)

        if detail_str:
            first_sentence = f'The strongest match here is "{title}" by {artist}, with a {detail_str} profile.'
        else:
            first_sentence = f'The strongest match here is "{title}" by {artist}.'

        if len(tracks) >= 2:
            alt = tracks[1]
            alt_title = alt.get("track_name", "another option")
            alt_artist = alt.get("artist_name", "another artist")
            second_sentence = f'I also included "{alt_title}" by {alt_artist} as a nearby option if you want a slightly different angle.'
        else:
            second_sentence = "I can keep going in this direction or tighten the recommendation if you want something more specific."

        return f"{first_sentence} {second_sentence}"

    def response_generation(self, sys_prompt: str, chat_history: list, recommend_item: str, max_new_tokens=512, response_format=None):
        return self._render_response(recommend_item)

    def batch_response_generation(self, sys_prompts: list[str], chat_histories: list[list], recommend_items: list[str], max_new_tokens=64):
        return [self._render_response(recommend_item) for recommend_item in recommend_items]
