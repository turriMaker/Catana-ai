"""
neural_player.py — Player that uses CatanNet to decide actions.
"""

import random
import torch
from catanatron.models.player import Player
from catanatron.models.enums import ActionType

from model import extract_features, build_action_mask, select_action


class NeuralPlayer(Player):
    """
    A Catan player driven by a CatanNet policy network.

    In training mode it records (log_prob) for each decision so the
    training loop can compute a REINFORCE loss at the end of the game.
    In eval mode it always picks the greedy (argmax) action.
    """

    def __init__(self, color, model, training=False):
        super().__init__(color)
        self.model    = model
        self.training = training
        self.log_probs = []   # filled during a game, cleared by reset_state()

    def decide(self, game, playable_actions):
        # DISCARD is not in the fixed action space — handle with random
        if all(a.action_type == ActionType.DISCARD for a in playable_actions):
            return random.choice(playable_actions)

        features     = extract_features(game.state, self.color)
        state_tensor = torch.tensor(features).unsqueeze(0)  # (1, STATE_SIZE)

        logits = self.model(state_tensor).squeeze(0)  # (ACTION_SPACE_SIZE,)

        mask, idx_to_action = build_action_mask(playable_actions)
        action, log_prob, _ = select_action(
            logits, mask, idx_to_action, playable_actions, training=self.training
        )

        if self.training and log_prob is not None:
            self.log_probs.append(log_prob)

        return action

    def reset_state(self):
        """Called by catanatron between games."""
        self.log_probs = []


class PPOPlayer(Player):
    """
    A Catan player for PPO training.

    Stores (state, action_idx, log_prob_old, mask) per step so the PPO
    update loop can recompute probabilities with the current policy.
    """

    def __init__(self, color, model, training=False):
        super().__init__(color)
        self.model    = model
        self.training = training
        self.states    = []   # state tensors  (STATE_SIZE,)
        self.actions   = []   # chosen action indices
        self.log_probs = []   # detached log_probs from the collection policy
        self.masks     = []   # bool tensors  (ACTION_SPACE_SIZE,)

    def decide(self, game, playable_actions):
        if all(a.action_type == ActionType.DISCARD for a in playable_actions):
            return random.choice(playable_actions)

        features     = extract_features(game.state, self.color)
        state_tensor = torch.tensor(features)

        with torch.no_grad():
            logits, _ = self.model(state_tensor.unsqueeze(0))
        logits = logits.squeeze(0)

        mask, idx_to_action = build_action_mask(playable_actions)
        action, log_prob, action_idx = select_action(
            logits, mask, idx_to_action, playable_actions, training=self.training
        )

        if self.training and log_prob is not None and action_idx is not None:
            self.states.append(state_tensor)
            self.actions.append(action_idx)
            self.log_probs.append(log_prob.detach())
            self.masks.append(torch.tensor(mask, dtype=torch.bool))

        return action

    def reset_state(self):
        self.states    = []
        self.actions   = []
        self.log_probs = []
        self.masks     = []
