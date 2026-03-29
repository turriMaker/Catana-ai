"""
model.py — Feature extraction, action encoding, and neural network architecture.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from itertools import combinations_with_replacement

from catanatron.models.enums import (
    RESOURCES, DEVELOPMENT_CARDS, ActionType,
    SETTLEMENT, CITY,
)
from catanatron.models.map import NUM_NODES, NUM_EDGES, NUM_TILES
from catanatron.models.board import base_map, get_edges

# ─── State feature constants ───────────────────────────────────────────────────

NUM_PLAYERS = 4

# Keys per player in state.player_state (prefixed with "P{idx}_")
PLAYER_KEYS = [
    "VICTORY_POINTS", "ROADS_AVAILABLE", "SETTLEMENTS_AVAILABLE",
    "CITIES_AVAILABLE", "HAS_ROAD", "HAS_ARMY", "HAS_ROLLED",
    "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN", "ACTUAL_VICTORY_POINTS", "LONGEST_ROAD_LENGTH",
    *[f"{r}_IN_HAND" for r in RESOURCES],        # 5 resources
    *[f"{d}_IN_HAND" for d in DEVELOPMENT_CARDS], # 5 dev cards
    *[f"PLAYED_{d}" for d in DEVELOPMENT_CARDS],  # 5 played dev cards
]
# 10 + 5 + 5 + 5 = 25 features per player, 4 players = 100

# Board edges in deterministic order
EDGES = get_edges()  # list of (node_a, node_b), len = 72
EDGE_TO_IDX = {}
for _i, _e in enumerate(EDGES):
    EDGE_TO_IDX[_e] = _i
    EDGE_TO_IDX[(_e[1], _e[0])] = _i  # also handle reversed edge

# Land tile coordinates in sorted order (for robber encoding)
LAND_COORDS = sorted(base_map.land_tiles.keys())  # 19 tiles
COORD_TO_IDX = {c: i for i, c in enumerate(LAND_COORDS)}

# State vector size = 100 + 54 + 72 + 19 + 5 = 250
STATE_SIZE = (NUM_PLAYERS * len(PLAYER_KEYS) + NUM_NODES + NUM_EDGES + NUM_TILES + len(RESOURCES))

# ─── Action space constants ────────────────────────────────────────────────────
#
# Layout (total = 249 actions):
#   [0]       ROLL
#   [1]       END_TURN
#   [2]       BUY_DEVELOPMENT_CARD
#   [3]       PLAY_KNIGHT_CARD
#   [4]       PLAY_ROAD_BUILDING
#   [5..58]   BUILD_SETTLEMENT  (54 nodes)
#   [59..112] BUILD_CITY        (54 nodes)
#   [113..184] BUILD_ROAD       (72 edges)
#   [185..203] MOVE_ROBBER      (19 tiles)
#   [204..208] PLAY_MONOPOLY    (5 resources)
#   [209..223] PLAY_YEAR_OF_PLENTY (15 combos with replacement, 2 resources)
#   [224..228] PLAY_YEAR_OF_PLENTY (5 single-resource, when bank is low)
#   [229..248] MARITIME_TRADE   (20 combos: 5 give × 4 receive, i≠j)

ACTION_SPACE_SIZE = 249

_S = 5  # offset after the 5 simple actions

OFFSET_SETTLEMENT   = _S
OFFSET_CITY         = _S + NUM_NODES
OFFSET_ROAD         = _S + 2 * NUM_NODES
OFFSET_ROBBER       = _S + 2 * NUM_NODES + NUM_EDGES
OFFSET_MONOPOLY     = _S + 2 * NUM_NODES + NUM_EDGES + NUM_TILES
OFFSET_YOP          = _S + 2 * NUM_NODES + NUM_EDGES + NUM_TILES + len(RESOURCES)
OFFSET_YOP1         = _S + 2 * NUM_NODES + NUM_EDGES + NUM_TILES + len(RESOURCES) + 15  # single-resource YOP
OFFSET_MARITIME     = _S + 2 * NUM_NODES + NUM_EDGES + NUM_TILES + len(RESOURCES) + 15 + len(RESOURCES)

# Year of Plenty: 15 combos with replacement from 5 resources (2-card), + 5 single-card
YOP_COMBOS = list(combinations_with_replacement(range(len(RESOURCES)), 2))
YOP_COMBO_IDX = {c: i for i, c in enumerate(YOP_COMBOS)}

# Maritime trade: give resource i, receive resource j (i != j) → 20 combos
MARITIME_COMBOS = [(i, j) for i in range(len(RESOURCES)) for j in range(len(RESOURCES)) if i != j]
MARITIME_COMBO_IDX = {c: i for i, c in enumerate(MARITIME_COMBOS)}

# ─── Feature extraction ────────────────────────────────────────────────────────

def extract_features(state, color):
    """
    Convert a catanatron State to a numpy float32 vector of size STATE_SIZE.
    The perspective is rotated so the given color is always 'player 0'.
    """
    features = []

    # 1. Player features — own player first, then others in seating order
    my_idx = state.color_to_index[color]
    num_players = len(state.colors)
    player_order = [(my_idx + i) % num_players for i in range(num_players)]
    # Pad to NUM_PLAYERS if fewer players in game
    while len(player_order) < NUM_PLAYERS:
        player_order.append(player_order[-1])

    for slot, p_idx in enumerate(player_order[:NUM_PLAYERS]):
        if slot < num_players:
            for key in PLAYER_KEYS:
                val = state.player_state.get(f"P{p_idx}_{key}", 0)
                features.append(float(val))
        else:
            features.extend([0.0] * len(PLAYER_KEYS))

    # 2. Board nodes — encoding per node:
    #    0=empty, 1=own settlement, 2=own city, -1=opp settlement, -2=opp city
    for node_id in range(NUM_NODES):
        if node_id in state.board.buildings:
            c, btype = state.board.buildings[node_id]
            if c == color:
                features.append(2.0 if btype == CITY else 1.0)
            else:
                features.append(-2.0 if btype == CITY else -1.0)
        else:
            features.append(0.0)

    # 3. Board edges — encoding per edge:
    #    0=no road, 1=own road, -1=opponent road
    for edge in EDGES:
        if edge in state.board.roads:
            c = state.board.roads[edge]
            features.append(1.0 if c == color else -1.0)
        else:
            features.append(0.0)

    # 4. Robber position — one-hot over 19 tiles
    robber_vec = [0.0] * NUM_TILES
    coord = state.board.robber_coordinate
    if coord in COORD_TO_IDX:
        robber_vec[COORD_TO_IDX[coord]] = 1.0
    features.extend(robber_vec)

    # 5. Bank resources — normalized by starting count (19 each)
    for i in range(len(RESOURCES)):
        features.append(state.resource_freqdeck[i] / 19.0)

    return np.array(features, dtype=np.float32)

# ─── Action encoding ───────────────────────────────────────────────────────────

def action_to_idx(action):
    """
    Map an Action to its index in the action space.
    Returns None for actions not in the fixed action space (e.g., DISCARD).
    """
    atype = action.action_type
    val   = action.value

    if atype == ActionType.ROLL:
        return 0
    if atype == ActionType.END_TURN:
        return 1
    if atype == ActionType.BUY_DEVELOPMENT_CARD:
        return 2
    if atype == ActionType.PLAY_KNIGHT_CARD:
        return 3
    if atype == ActionType.PLAY_ROAD_BUILDING:
        return 4
    if atype == ActionType.BUILD_SETTLEMENT:
        return OFFSET_SETTLEMENT + val
    if atype == ActionType.BUILD_CITY:
        return OFFSET_CITY + val
    if atype == ActionType.BUILD_ROAD:
        edge_idx = EDGE_TO_IDX.get(val)
        if edge_idx is None:
            return None
        return OFFSET_ROAD + edge_idx
    if atype == ActionType.MOVE_ROBBER:
        coord = val[0]
        tile_idx = COORD_TO_IDX.get(coord)
        if tile_idx is None:
            return None
        return OFFSET_ROBBER + tile_idx
    if atype == ActionType.PLAY_MONOPOLY:
        if val not in RESOURCES:
            return None
        return OFFSET_MONOPOLY + RESOURCES.index(val)
    if atype == ActionType.PLAY_YEAR_OF_PLENTY:
        if len(val) == 1:
            r = val[0]
            if r not in RESOURCES:
                return None
            return OFFSET_YOP1 + RESOURCES.index(r)
        r1, r2 = val
        combo = tuple(sorted([RESOURCES.index(r1), RESOURCES.index(r2)]))
        idx = YOP_COMBO_IDX.get(combo)
        if idx is None:
            return None
        return OFFSET_YOP + idx
    if atype == ActionType.MARITIME_TRADE:
        # val is a 5-tuple: (give, give, give[, give], get)
        # First element is what we give, last is what we receive
        give = val[0]
        get  = val[-1]
        if give not in RESOURCES or get not in RESOURCES:
            return None
        combo = (RESOURCES.index(give), RESOURCES.index(get))
        idx = MARITIME_COMBO_IDX.get(combo)
        if idx is None:
            return None
        return OFFSET_MARITIME + idx
    return None


def build_action_mask(playable_actions):
    """
    Build a boolean mask over ACTION_SPACE_SIZE and a mapping from
    action index → Action for all encodable actions.
    """
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    idx_to_action = {}

    for action in playable_actions:
        idx = action_to_idx(action)
        if idx is not None and 0 <= idx < ACTION_SPACE_SIZE:
            mask[idx] = True
            idx_to_action[idx] = action

    return mask, idx_to_action


def select_action(logits, mask, idx_to_action, playable_actions, training=True):
    """
    Sample or pick an action from network logits using a validity mask.

    Returns:
        action       — the chosen Action object
        log_prob     — log probability tensor (with grad) or None
        chosen_idx   — the index chosen in the action space
    """
    mask_tensor = torch.tensor(mask, dtype=torch.bool, device=logits.device)

    if mask_tensor.any():
        masked_logits = logits.clone()
        masked_logits[~mask_tensor] = float('-inf')
        probs = torch.softmax(masked_logits, dim=-1)
        dist  = torch.distributions.Categorical(probs)

        if training:
            idx = dist.sample()
        else:
            idx = probs.argmax()

        log_prob = dist.log_prob(idx)
        idx_val  = idx.item()

        if idx_val in idx_to_action:
            return idx_to_action[idx_val], log_prob, idx_val

    # Fallback: random (happens for DISCARD or unencodable actions)
    return random.choice(playable_actions), None, None

# ─── Neural network ────────────────────────────────────────────────────────────

class CatanNet(nn.Module):
    """
    Policy network for Catan.

    Input:  state vector of size STATE_SIZE (250)
    Output: logits over ACTION_SPACE_SIZE (244) actions
    """

    def __init__(self, hidden_size=256, num_layers=3):
        super().__init__()

        layers = [nn.Linear(STATE_SIZE, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, ACTION_SPACE_SIZE))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CatanActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    Shared trunk → policy head (logits over ACTION_SPACE_SIZE)
                 → value head  (scalar V(s))

    Compatible with CatanNet checkpoints via load_from_reinforce().
    """

    def __init__(self, hidden_size=256, num_layers=3):
        super().__init__()
        self.num_layers  = num_layers
        self.hidden_size = hidden_size

        trunk_layers = [nn.Linear(STATE_SIZE, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            trunk_layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        self.trunk       = nn.Sequential(*trunk_layers)
        self.policy_head = nn.Linear(hidden_size, ACTION_SPACE_SIZE)
        self.value_head  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """Returns (logits, value) — shapes (B, ACTION_SPACE_SIZE) and (B,)."""
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def load_from_reinforce(self, reinforce_state_dict):
        """
        Load trunk + policy head from a CatanNet (REINFORCE) checkpoint.
        Value head is left randomly initialised.

        CatanNet.net layout for num_layers=3:
          net.0 Linear(STATE_SIZE, hidden)   ← trunk
          net.1 ReLU                         ← trunk
          net.2 Linear(hidden, hidden)       ← trunk
          net.3 ReLU                         ← trunk
          net.4 Linear(hidden, hidden)       ← trunk
          net.5 ReLU                         ← trunk
          net.6 Linear(hidden, ACTION_SIZE)  ← policy_head
        """
        num_trunk = 2 * self.num_layers  # = 6 for num_layers=3
        new_dict = {}
        for k, v in reinforce_state_dict.items():
            parts = k.split('.')          # ['net', '6', 'weight']
            layer_idx = int(parts[1])
            rest = '.'.join(parts[2:])    # 'weight' or 'bias'
            if layer_idx < num_trunk:
                new_dict[f'trunk.{layer_idx}.{rest}'] = v
            else:
                new_dict[f'policy_head.{rest}'] = v
        # strict=False: value_head keys are missing — stays random
        self.load_state_dict(new_dict, strict=False)
