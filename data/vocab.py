"""
Event vocabulary for behavioral sequence tokenization.

Each in-game event is mapped to a discrete token ID, analogous to word tokens in NLP.
The key insight is that we're NOT doing text NLP — we're applying transformer attention
to behavioral event sequences with custom tokenization and game-time positional encoding.
"""

EVENT_VOCAB = {
    # Combat events
    "KILL": 0,
    "DEATH": 1,
    "ASSIST": 2,
    "MULTI_KILL": 3,           # 2+ kills within 15 seconds
    "DEATH_STREAK": 4,          # 3+ deaths without a kill

    # Economy events
    "BIG_PURCHASE": 5,          # item costing > 2000 gold
    "SMALL_PURCHASE": 6,        # item costing < 500 gold
    "GOLD_SPIKE_UP": 7,         # gold increased > 500 in 1 min
    "GOLD_SPIKE_DOWN": 8,       # gold dropped > 500 in 1 min (died with gold)

    # Performance signals
    "LH_ABOVE_AVG": 9,          # last hits above rolling average
    "LH_BELOW_AVG": 10,         # last hits below rolling average
    "XP_FALLING_BEHIND": 11,    # XP significantly below team average

    # Tempo / engagement signals
    "ACTION_BURST": 12,         # action rate > 2 std devs above mean
    "ACTION_DROUGHT": 13,       # action rate < 0.5 std devs below mean
    "LONG_IDLE": 14,            # no meaningful action for > 30 seconds

    # Team context
    "TEAM_FIGHT_WIN": 15,       # team won a fight (3+ kills within 20 sec)
    "TEAM_FIGHT_LOSS": 16,      # team lost a fight
    "TOWER_LOST": 17,           # team lost a tower
    "TOWER_TAKEN": 18,          # team took a tower

    # Meta tokens
    "[PAD]": 19,
    "[CLS]": 20,                # start of sequence
    "[SEP]": 21,                # minute boundary separator
}

VOCAB_SIZE = len(EVENT_VOCAB)
PAD_TOKEN_ID = EVENT_VOCAB["[PAD]"]
CLS_TOKEN_ID = EVENT_VOCAB["[CLS]"]
SEP_TOKEN_ID = EVENT_VOCAB["[SEP]"]

# Reverse mapping for interpretability
ID_TO_EVENT = {v: k for k, v in EVENT_VOCAB.items()}

# Number of continuous features per token
NUM_CONTINUOUS_FEATURES = 6  # gold_diff, xp_diff, kda_ratio, team_gold_diff, net_worth_rank, action_rate
