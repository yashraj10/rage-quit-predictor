"""
Kafka Producer for Rage Quit Predictor.

Reads match data from OpenDota API and publishes behavioral events
to the 'match-events' Kafka topic with match_id as partition key.

Topic: match-events
Partitions: 4 (match_id % 4)
Schema: {match_id, player_slot, event_type, timestamp, gold_diff, xp_diff, kda}
"""

import json
import time
import random
import logging
from kafka import KafkaProducer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOPIC = "match-events"
BOOTSTRAP_SERVERS = "localhost:9092"

EVENT_TYPES = [
    "KILL", "DEATH", "ASSIST", "MULTI_KILL", "DEATH_STREAK",
    "BIG_PURCHASE", "SMALL_PURCHASE", "GOLD_SPIKE_UP", "GOLD_SPIKE_DOWN",
    "LH_ABOVE_AVG", "LH_BELOW_AVG", "XP_FALLING_BEHIND",
    "ACTION_BURST", "ACTION_DROUGHT", "LONG_IDLE",
    "TEAM_FIGHT_WIN", "TEAM_FIGHT_LOSS", "TOWER_LOST", "TOWER_TAKEN"
]


def create_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
    )


def simulate_match_events(match_id: int, num_events: int = 20) -> list[dict]:
    """Simulate behavioral events for a match (mimics OpenDota API output)."""
    events = []
    for player_slot in range(10):
        for i in range(num_events // 10):
            event = {
                "match_id": match_id,
                "player_slot": player_slot,
                "event_type": random.choice(EVENT_TYPES),
                "timestamp": int(time.time()) + i * 60,
                "gold_diff": random.uniform(-2000, 2000),
                "xp_diff": random.uniform(-1500, 1500),
                "kda": round(random.uniform(0, 10), 2),
                "minute": i,
            }
            events.append(event)
    return events


def produce_match_events(num_matches: int = 5):
    """Produce events for multiple matches to Kafka."""
    producer = create_producer()
    total_sent = 0

    logger.info(f"Starting producer — sending {num_matches} matches to topic '{TOPIC}'")

    for i in range(num_matches):
        match_id = random.randint(8000000, 9000000)
        events = simulate_match_events(match_id)

        for event in events:
            producer.send(
                topic=TOPIC,
                key=match_id,
                value=event,
            )
            total_sent += 1

        logger.info(f"Match {match_id}: sent {len(events)} events (partition {match_id % 4})")
        time.sleep(0.1)

    producer.flush()
    logger.info(f"✅ Done. Total events sent: {total_sent}")
    return total_sent


if __name__ == "__main__":
    produce_match_events(num_matches=5)