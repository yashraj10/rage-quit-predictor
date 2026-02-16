"""
Data processing pipeline: raw OpenDota match JSON → behavioral event sequences.

This is the core feature engineering step. Each player-match is converted into
a chronological sequence of discrete behavioral events (tokens) with associated
continuous features, suitable for transformer input.

Usage:
    python -m data.process --input_dir data/raw --output_path data/processed/sequences.pkl
"""

import argparse
import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from data.vocab import (
    CLS_TOKEN_ID,
    EVENT_VOCAB,
    NUM_CONTINUOUS_FEATURES,
    SEP_TOKEN_ID,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Item cost thresholds (approximate Dota 2 item costs)
BIG_ITEM_COST = 2000
SMALL_ITEM_COST = 500

# Approximate item costs — covers common items. Unknown items default to medium cost.
ITEM_COSTS = {
    "tango": 90, "clarity": 50, "salve": 100, "ward_observer": 0,
    "ward_sentry": 50, "magic_stick": 200, "boots": 500,
    "power_treads": 1400, "phase_boots": 1500, "blink": 2250,
    "force_staff": 2175, "black_king_bar": 4050, "desolator": 3500,
    "butterfly": 4975, "heart": 5200, "rapier": 6200, "monkey_king_bar": 4975,
    "battle_fury": 4100, "manta": 4600, "sange_and_yasha": 4100,
    "aghanims_scepter": 4200, "refresher": 5200, "assault": 5250,
    "radiance": 5150, "dagon": 2700, "orchid": 3475, "bloodthorn": 6825,
}


def get_item_cost(item_name: str) -> int:
    """Look up approximate item cost. Returns 1000 for unknown items."""
    return ITEM_COSTS.get(item_name, 1000)


def is_radiant(player_slot: int) -> bool:
    """Determine if a player is on the Radiant side based on player_slot."""
    return player_slot < 128


class MatchProcessor:
    """Processes a single match JSON into player behavioral sequences."""

    def __init__(self, max_seq_len: int = 256):
        self.max_seq_len = max_seq_len

    def process_match(self, match_data: dict) -> list[dict]:
        """
        Process a full match into a list of player sequence records.

        Returns:
            List of dicts, one per player, each containing:
                - event_ids: list[int]
                - continuous: list[list[float]]
                - minutes: list[int]
                - label: int (0 or 1)
                - match_id: int
                - player_slot: int
                - hero_id: int
        """
        match_id = match_data.get("match_id", 0)
        duration = match_data.get("duration", 0)
        radiant_win = match_data.get("radiant_win", False)
        duration_minutes = duration // 60

        players = match_data.get("players", [])
        if len(players) != 10:
            return []

        # Compute team-level stats for context
        team_stats = self._compute_team_stats(players, duration_minutes)

        records = []
        for player in players:
            try:
                record = self._process_player(
                    player, match_id, duration_minutes, radiant_win, team_stats
                )
                if record is not None:
                    records.append(record)
            except Exception as e:
                logger.debug(f"Error processing player in match {match_id}: {e}")
                continue

        return records

    def _compute_team_stats(
        self, players: list[dict], duration_minutes: int
    ) -> dict:
        """Compute per-minute team-level aggregate stats."""
        radiant_gold = defaultdict(float)
        dire_gold = defaultdict(float)
        radiant_xp = defaultdict(float)
        dire_xp = defaultdict(float)
        radiant_count = 0
        dire_count = 0

        for p in players:
            gold_t = p.get("gold_t") or []
            xp_t = p.get("xp_t") or []
            slot = p.get("player_slot", 0)
            is_rad = is_radiant(slot)

            if is_rad:
                radiant_count += 1
            else:
                dire_count += 1

            for minute in range(min(len(gold_t), duration_minutes + 1)):
                if is_rad:
                    radiant_gold[minute] += gold_t[minute]
                    radiant_xp[minute] += xp_t[minute] if minute < len(xp_t) else 0
                else:
                    dire_gold[minute] += gold_t[minute]
                    dire_xp[minute] += xp_t[minute] if minute < len(xp_t) else 0

        return {
            "radiant_gold": dict(radiant_gold),
            "dire_gold": dict(dire_gold),
            "radiant_xp": dict(radiant_xp),
            "dire_xp": dict(dire_xp),
        }
        
    def _determine_label(
        self, player: dict, duration_minutes: int, radiant_win: bool
    ) -> int:
        """
        Determine rage quit label.
        A player is a rage quit if leaver_status >= 2 (abandoned or AFK).
        """
        leaver_status = player.get("leaver_status", 0)
        if leaver_status >= 2:
            return 1
        return 0 
        """
        Determine rage quit label.

        Criteria:
            - leaver_status >= 2 (abandoned or AFK)
            - Likely left early (not at game end)
            - Team was losing
        """
        leaver_status = player.get("leaver_status", 0)
        if leaver_status < 2:
            return 0

        # Check if player's team was losing
        slot = player.get("player_slot", 0)
        player_is_radiant = is_radiant(slot)
        team_won = (player_is_radiant and radiant_win) or (
            not player_is_radiant and not radiant_win
        )

        # If they abandoned but their team won, less likely rage quit
        if team_won:
            return 0

        return 1

    def _process_player(
        self,
        player: dict,
        match_id: int,
        duration_minutes: int,
        radiant_win: bool,
        team_stats: dict,
    ) -> dict | None:
        """Process a single player into a behavioral event sequence."""
        gold_t = player.get("gold_t") or []
        xp_t = player.get("xp_t") or []
        lh_t = player.get("lh_t") or []

        if not gold_t or not xp_t:
            return None

        slot = player.get("player_slot", 0)
        player_is_radiant = is_radiant(slot)
        label = self._determine_label(player, duration_minutes, radiant_win)

        # Build minute-by-minute events
        events_by_minute = defaultdict(list)  # minute -> list of (event_id, features)

        # --- Extract kill/death events ---
        kills_log = player.get("kills_log") or []
        kill_times = []
        for kill in kills_log:
            minute = (kill.get("time") or 0) // 60
            if minute >= 0:
                kill_times.append(minute)
                events_by_minute[minute].append(EVENT_VOCAB["KILL"])

        # Deaths: inferred from gold drops + kills_log gaps
        # We use the difference in gold_t as a proxy for deaths
        death_count = 0
        kills_since_last_death = len(kill_times)  # track for death streaks
        for minute in range(1, min(len(gold_t), duration_minutes + 1)):
            gold_diff = gold_t[minute] - gold_t[minute - 1]

            # Large gold drop usually indicates a death (lost unreliable gold)
            if gold_diff < -200:
                events_by_minute[minute].append(EVENT_VOCAB["DEATH"])
                death_count += 1
                kills_since_last_death = 0

                # Check for death streak
                if death_count >= 3 and kills_since_last_death == 0:
                    events_by_minute[minute].append(EVENT_VOCAB["DEATH_STREAK"])

                # Gold spike down
                if gold_diff < -500:
                    events_by_minute[minute].append(EVENT_VOCAB["GOLD_SPIKE_DOWN"])
            else:
                if minute in kill_times:
                    kills_since_last_death += 1
                    if kills_since_last_death > 0:
                        death_count = 0  # reset death streak on kill

                # Gold spike up
                if gold_diff > 500:
                    events_by_minute[minute].append(EVENT_VOCAB["GOLD_SPIKE_UP"])

        # --- Assist events (from kills_log of teammates — approximate) ---
        assists = player.get("assists", 0)
        if assists > 0 and duration_minutes > 0:
            # Distribute assists roughly evenly (we don't have exact timestamps)
            assist_interval = max(1, duration_minutes // max(assists, 1))
            for m in range(assist_interval, duration_minutes, assist_interval):
                events_by_minute[m].append(EVENT_VOCAB["ASSIST"])

        # --- Purchase events ---
        purchase_log = player.get("purchase_log") or []
        for purchase in purchase_log:
            minute = (purchase.get("time") or 0) // 60
            item_name = purchase.get("key", "")
            cost = get_item_cost(item_name)
            if cost >= BIG_ITEM_COST:
                events_by_minute[minute].append(EVENT_VOCAB["BIG_PURCHASE"])
            elif cost <= SMALL_ITEM_COST:
                events_by_minute[minute].append(EVENT_VOCAB["SMALL_PURCHASE"])

        # --- Last hit performance ---
        if len(lh_t) > 3:
            rolling_avg = []
            for i in range(len(lh_t)):
                window = lh_t[max(0, i - 3): i + 1]
                avg_gain = np.mean(np.diff(window)) if len(window) > 1 else 0
                rolling_avg.append(avg_gain)

            for minute in range(1, min(len(lh_t), duration_minutes + 1)):
                current_gain = lh_t[minute] - lh_t[minute - 1]
                if minute < len(rolling_avg) and rolling_avg[minute] > 0:
                    if current_gain > rolling_avg[minute] * 1.3:
                        events_by_minute[minute].append(EVENT_VOCAB["LH_ABOVE_AVG"])
                    elif current_gain < rolling_avg[minute] * 0.5:
                        events_by_minute[minute].append(EVENT_VOCAB["LH_BELOW_AVG"])

        # --- XP falling behind ---
        for minute in range(len(xp_t)):
            team_key = "radiant_xp" if player_is_radiant else "dire_xp"
            team_total_xp = team_stats[team_key].get(minute, 0)
            team_avg_xp = team_total_xp / 5 if team_total_xp > 0 else 0
            if team_avg_xp > 0 and xp_t[minute] < team_avg_xp * 0.7:
                events_by_minute[minute].append(EVENT_VOCAB["XP_FALLING_BEHIND"])

        # --- Action rate signals (approximate from actions dict) ---
        actions = player.get("actions") or {}
        total_actions = sum(actions.values())
        avg_actions_per_min = total_actions / max(duration_minutes, 1)
        std_actions = avg_actions_per_min * 0.3  # rough estimate

        # We don't have per-minute action data, so we use lh_t changes as a proxy
        for minute in range(1, min(len(lh_t), duration_minutes + 1)):
            activity = lh_t[minute] - lh_t[minute - 1]  # proxy for activity
            if activity == 0 and minute > 2:
                events_by_minute[minute].append(EVENT_VOCAB["ACTION_DROUGHT"])
            elif activity > 15:  # very high cs in one minute
                events_by_minute[minute].append(EVENT_VOCAB["ACTION_BURST"])

        # --- Team fight approximation (from objectives/teamfights if available) ---
        teamfights = []  # Would come from match_data.get("teamfights", [])
        # For now, we approximate from kill clustering in gold_t patterns
        # This is a simplification — with the full teamfights API field, this is richer

        # --- Build the final sequence ---
        event_ids = [CLS_TOKEN_ID]
        continuous = [[0.0] * NUM_CONTINUOUS_FEATURES]
        minutes_list = [0]

        for minute in range(min(duration_minutes + 1, 80)):
            minute_events = events_by_minute.get(minute, [])

            # If no events this minute, still add separator
            if not minute_events and minute > 0:
                event_ids.append(SEP_TOKEN_ID)
                continuous.append(self._get_continuous_features(
                    player, minute, team_stats, player_is_radiant
                ))
                minutes_list.append(min(minute, 89))
                continue

            for evt in minute_events:
                event_ids.append(evt)
                continuous.append(self._get_continuous_features(
                    player, minute, team_stats, player_is_radiant
                ))
                minutes_list.append(min(minute, 89))

            # Add separator after each minute's events
            if minute < duration_minutes:
                event_ids.append(SEP_TOKEN_ID)
                continuous.append(self._get_continuous_features(
                    player, minute, team_stats, player_is_radiant
                ))
                minutes_list.append(min(minute, 89))

        # Truncate to max_seq_len
        event_ids = event_ids[: self.max_seq_len]
        continuous = continuous[: self.max_seq_len]
        minutes_list = minutes_list[: self.max_seq_len]

        if len(event_ids) < 10:
            return None  # Too short to be useful

        return {
            "event_ids": event_ids,
            "continuous": continuous,
            "minutes": minutes_list,
            "label": label,
            "match_id": match_id,
            "player_slot": slot,
            "hero_id": player.get("hero_id", 0),
        }

    def _get_continuous_features(
        self,
        player: dict,
        minute: int,
        team_stats: dict,
        player_is_radiant: bool,
    ) -> list[float]:
        """Compute continuous features for a given minute."""
        gold_t = player.get("gold_t") or []
        xp_t = player.get("xp_t") or []

        player_gold = gold_t[minute] if minute < len(gold_t) else 0
        player_xp = xp_t[minute] if minute < len(xp_t) else 0

        # Gold diff vs opposing team average
        opp_key = "dire_gold" if player_is_radiant else "radiant_gold"
        opp_team_gold = team_stats[opp_key].get(minute, 0)
        opp_avg_gold = opp_team_gold / 5 if opp_team_gold > 0 else 1
        gold_diff = (player_gold - opp_avg_gold) / max(opp_avg_gold, 1)

        # XP diff vs opposing team average
        opp_xp_key = "dire_xp" if player_is_radiant else "radiant_xp"
        opp_team_xp = team_stats[opp_xp_key].get(minute, 0)
        opp_avg_xp = opp_team_xp / 5 if opp_team_xp > 0 else 1
        xp_diff = (player_xp - opp_avg_xp) / max(opp_avg_xp, 1)

        # KDA ratio (running — approximate)
        kills = player.get("kills", 0)
        deaths = player.get("deaths", 1)
        assists = player.get("assists", 0)
        kda = (kills + assists) / max(deaths, 1)

        # Team gold diff
        team_gold_key = "radiant_gold" if player_is_radiant else "dire_gold"
        team_gold = team_stats[team_gold_key].get(minute, 0)
        team_gold_diff = (team_gold - opp_team_gold) / max(opp_team_gold, 1)

        # Net worth rank (approximate — use gold as proxy, normalized 0-1)
        net_worth_rank = min(gold_diff + 0.5, 1.0)  # rough normalization

        # Action rate proxy
        lh_t = player.get("lh_t") or []
        if minute > 0 and minute < len(lh_t):
            action_rate = (lh_t[minute] - lh_t[minute - 1]) / 10.0  # normalize
        else:
            action_rate = 0.0

        return [
            float(np.clip(gold_diff, -3, 3)),
            float(np.clip(xp_diff, -3, 3)),
            float(np.clip(kda, 0, 10)),
            float(np.clip(team_gold_diff, -3, 3)),
            float(np.clip(net_worth_rank, 0, 1)),
            float(np.clip(action_rate, 0, 3)),
        ]


def process_all_matches(input_dir: str, output_path: str, max_seq_len: int = 256):
    """Process all raw match JSONs into sequences and save."""
    input_path = Path(input_dir)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    processor = MatchProcessor(max_seq_len=max_seq_len)
    all_records = []
    match_files = list(input_path.glob("*.json"))
    match_files = [f for f in match_files if f.name != "checkpoint.json"]

    logger.info(f"Processing {len(match_files)} match files...")

    for i, match_file in enumerate(match_files):
        try:
            with open(match_file) as f:
                match_data = json.load(f)
            records = processor.process_match(match_data)
            all_records.extend(records)
        except Exception as e:
            logger.debug(f"Error processing {match_file.name}: {e}")
            continue

        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i+1}/{len(match_files)} matches, {len(all_records)} sequences")

    # Summary statistics
    labels = [r["label"] for r in all_records]
    rage_quit_rate = sum(labels) / len(labels) if labels else 0

    logger.info(f"Total sequences: {len(all_records)}")
    logger.info(f"Rage quit rate: {rage_quit_rate:.2%}")
    logger.info(f"Positive samples: {sum(labels)}, Negative samples: {len(labels) - sum(labels)}")

    with open(output_file, "wb") as f:
        pickle.dump(all_records, f)

    logger.info(f"Saved to {output_file}")
    return all_records


def main():
    parser = argparse.ArgumentParser(description="Process raw match data into sequences")
    parser.add_argument("--input_dir", type=str, default="data/raw")
    parser.add_argument("--output_path", type=str, default="data/processed/sequences.pkl")
    parser.add_argument("--max_seq_len", type=int, default=256)
    args = parser.parse_args()

    process_all_matches(args.input_dir, args.output_path, args.max_seq_len)


if __name__ == "__main__":
    main()
