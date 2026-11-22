#!/usr/bin/env python3
"""Reinforcement-learning agent for Hamurabi using SimpleAI abstractions.

This module builds a Q-learning (reinforcement-based) agent that learns to play
the Hamurabi implementation in ``hamurabi.py`` by interacting with it over many
episodes.

Training schedule (as requested):
- First 100 training iterations/episodes: high exploration to seek broad
  coverage of game states (epsilon-greedy with epsilon=1.0, i.e. random policy).
- After 100 iterations: transition to a conservative/exploitative policy by
  decaying epsilon toward a low value and following learned Q-values.

SimpleAI is used to model the per-turn decision space via a
``simpleai.search.models.SearchProblem`` subclass (``HamurabiTurnProblem``),
which provides the candidate action set for each observed state.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from simpleai.search.models import SearchProblem

from hamurabi import HamurabiGame, GameState


Action = Tuple[int, int, int, int]  # (acres_bought, acres_sold, bushels_fed, acres_planted)


@dataclass
class DiscretizationConfig:
    pop_bins: Tuple[float, ...] = (40, 80, 120, 170)
    bpp_bins: Tuple[float, ...] = (10, 20, 30, 45)
    app_bins: Tuple[float, ...] = (5, 8, 10, 13)
    price_bins: Tuple[int, ...] = (19, 22)


@dataclass
class RewardConfig:
    impeach_penalty: float = 120.0
    wipeout_penalty: float = 120.0
    survival_bonus: float = 2.0
    starve_penalty_scale: float = 40.0
    no_starve_bonus: float = 4.0
    low_starve_bonus: float = 1.0
    low_starve_ratio_threshold: float = 0.1
    reserves_low_threshold: float = 10.0
    reserves_low_penalty: float = 1.5
    reserves_high_threshold: float = 20.0
    reserves_high_bonus: float = 0.8
    app_good_low: float = 8.0
    app_good_high: float = 15.0
    app_good_bonus: float = 0.5
    app_low_threshold: float = 6.0
    app_low_penalty: float = 0.5
    completion_tier1_bonus: float = 80.0
    completion_tier1_avg_starv: float = 3.0
    completion_tier1_app: float = 10.0
    completion_tier2_bonus: float = 40.0
    completion_tier2_avg_starv: float = 10.0
    completion_tier2_app: float = 9.0
    completion_tier3_bonus: float = 15.0


@dataclass
class ActionGenConfig:
    feed_fracs: Tuple[float, ...] = (0.0, 0.4, 0.6, 0.8, 1.0, 1.1)
    sell_fracs: Tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)
    buy_fracs: Tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)
    plant_fracs: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass
class ConservativeScoreConfig:
    feed_weight: float = 6.0
    full_feed_bonus: float = 2.0
    plant_weight: float = 2.0
    sell_half_penalty: float = 1.0
    sell_half_threshold: float = 0.5
    buy_expensive_penalty: float = 0.5
    buy_expensive_price: int = 24
    buy_expensive_reserve_mult: float = 2.0


@dataclass
class ExplorationConfig:
    viable_feed_ratio: float = 1.0
    viable_fallback_ratio: float = 0.8
    viable_pool_prob: float = 0.8
    heuristic_jitter: float = 0.5


DEFAULT_DISCRETIZATION_CONFIG = DiscretizationConfig()
DEFAULT_REWARD_CONFIG = RewardConfig()
DEFAULT_ACTION_GEN_CONFIG = ActionGenConfig()
DEFAULT_CONSERVATIVE_SCORE_CONFIG = ConservativeScoreConfig()
DEFAULT_EXPLORATION_CONFIG = ExplorationConfig()


@dataclass(frozen=True)
class Observation:
    """Information available to the agent at the start of a year."""

    year: int
    population: int
    bushels: int
    acres: int
    land_price: int


def _bucketize(value: float, bins: Sequence[float]) -> int:
    """Return bucket index in 0..len(bins) based on ascending threshold bins."""
    for i, b in enumerate(bins):
        if value < b:
            return i
    return len(bins)


def discretize_state(obs: Observation, config: Optional[DiscretizationConfig] = None) -> Tuple[int, int, int, int, int]:
    """Convert a continuous/large game observation into a hashable state bucket.

    Buckets are chosen to capture strategically relevant distinctions while
    keeping the Q-table tractable.
    """
    if config is None:
        config = DEFAULT_DISCRETIZATION_CONFIG

    year = max(1, min(obs.year, HamurabiGame.MAX_YEARS))

    if obs.population <= 0:
        pop_b = 0
    else:
        pop_b = 1 + _bucketize(obs.population, config.pop_bins)

    bushels_per_pop = obs.bushels / max(1, obs.population)
    bpp_b = _bucketize(bushels_per_pop, config.bpp_bins)

    acres_per_pop = obs.acres / max(1, obs.population)
    app_b = _bucketize(acres_per_pop, config.app_bins)

    # Land price 17-26 -> 3 buckets (cheap/medium/expensive)
    # Preserve original semantics: <=19 cheap, <=22 medium, else expensive.
    if tuple(config.price_bins) == (19, 22):
        if obs.land_price <= 19:
            price_b = 0
        elif obs.land_price <= 22:
            price_b = 1
        else:
            price_b = 2
    else:
        # For custom bins, use <= threshold semantics for integer prices.
        price_b = len(config.price_bins)
        for i, b in enumerate(config.price_bins):
            if obs.land_price <= b:
                price_b = i
                break

    return (year, pop_b, bpp_b, app_b, price_b)


class HamurabiTurnProblem(SearchProblem):
    """SimpleAI SearchProblem representing candidate actions for one turn.

    The "state" held by this problem is the current ``Observation``. Actions are
    valid (buy/sell, feed, plant) tuples that the agent may choose from. The
    problem is used as a structured action generator/provider (via SimpleAI's
    ``SearchProblem`` abstraction) rather than as a graph-search task.
    """

    def __init__(
        self,
        game: HamurabiGame,
        observation: Observation,
        action_gen_config: Optional[ActionGenConfig] = None,
    ):
        self.game = game
        self.observation = observation
        self.action_gen_config = action_gen_config or DEFAULT_ACTION_GEN_CONFIG
        super().__init__(initial_state=observation)

    def actions(self, state: Observation) -> List[Action]:
        return generate_candidate_actions(self.game, state, self.action_gen_config)

    def result(self, state: Observation, action: Action) -> Observation:
        # Single-step decision problem; applying an action conceptually leads to
        # a terminal decision state for this turn.
        return state

    def is_goal(self, state: Observation) -> bool:
        return False

    def value(self, state: Observation):  # for local search compatibility
        return 0


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def generate_candidate_actions(
    game: HamurabiGame, obs: Observation, config: Optional[ActionGenConfig] = None
) -> List[Action]:
    """Generate a diverse set of valid candidate actions for the given state.

    Actions include conservative, balanced, and risky/aggressive choices to
    support exploration and learning.
    """
    if config is None:
        config = DEFAULT_ACTION_GEN_CONFIG

    s_pop = obs.population
    s_bushels = obs.bushels
    s_acres = obs.acres
    price = obs.land_price

    needed_food = s_pop * HamurabiGame.BUSHELS_PER_PERSON
    max_farmable = min(s_acres, s_pop * HamurabiGame.ACRES_PER_PERSON)

    # Feeding options: fractions of needed food (clamped to available later)
    feed_options = sorted(
        {
            _clamp(int(round(needed_food * f)), 0, s_bushels)
            for f in config.feed_fracs
        }
    )

    # Trade options expressed as target acreage changes relative to current.
    # Negative = sell fraction of acres, positive = buy fraction of affordable.
    trade_options: List[Tuple[int, int]] = [(0, 0)]
    for f in config.sell_fracs:
        sell = _clamp(int(round(s_acres * f)), 0, s_acres)
        if sell > 0:
            trade_options.append((0, sell))
    max_affordable = s_bushels // max(1, price)
    for f in config.buy_fracs:
        buy = _clamp(int(round(max_affordable * f)), 0, max_affordable)
        if buy > 0:
            trade_options.append((buy, 0))

    candidates: List[Action] = []
    seen = set()

    for buy, sell in trade_options:
        # Simulate resources after trade for feasibility of feed/plant
        bushels_after_trade = s_bushels - buy * price + sell * price
        acres_after_trade = s_acres + buy - sell
        if bushels_after_trade < 0 or acres_after_trade < 0:
            continue

        farm_cap = min(acres_after_trade, s_pop * HamurabiGame.ACRES_PER_PERSON)

        for feed in feed_options:
            feed = _clamp(feed, 0, bushels_after_trade)
            bushels_after_feed = bushels_after_trade - feed

            # Plant options: fractions of feasible (by land, labor, seed)
            max_by_seed = bushels_after_feed * 2
            feasible_plant = min(farm_cap, max_by_seed)
            for pf in config.plant_fracs:
                plant = _clamp(int(round(feasible_plant * pf)), 0, feasible_plant)
                action = (buy, sell, feed, plant)
                if action in seen:
                    continue
                ok, _ = game.validate_turn(buy, sell, feed, plant, price)
                if ok:
                    seen.add(action)
                    candidates.append(action)

    # Always ensure at least one valid action (do nothing / minimal safe fallback)
    if not candidates:
        # try minimal safe action
        for feed in (min(needed_food, s_bushels), 0):
            for plant in (0,):
                action = (0, 0, feed, plant)
                ok, _ = game.validate_turn(0, 0, feed, plant, price)
                if ok:
                    candidates.append(action)
        if not candidates:
            candidates.append((0, 0, 0, 0))

    return candidates


class QLearningHamurabiAgent:
    """Tabular Q-learning agent for Hamurabi.

    - Uses epsilon-greedy action selection.
    - For the first ``explore_iterations`` episodes, epsilon is 1.0 to broadly
      explore game states. Exploration is biased to still reach diverse
      (including late-game) states by preferring viable, less-tried actions.
    - After that, epsilon decays toward ``min_epsilon`` to exploit a
      conservative/best learned path.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 0.98,
        explore_iterations: int = 100,
        min_epsilon: float = 0.02,
        epsilon_decay: float = 0.95,
        seed: Optional[int] = None,
        discretization_config: Optional[DiscretizationConfig] = None,
        action_gen_config: Optional[ActionGenConfig] = None,
        conservative_score_config: Optional[ConservativeScoreConfig] = None,
        exploration_config: Optional[ExplorationConfig] = None,
        reward_config: Optional[RewardConfig] = None,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.explore_iterations = explore_iterations
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        self.discretization_config = discretization_config or copy.deepcopy(DEFAULT_DISCRETIZATION_CONFIG)
        self.action_gen_config = action_gen_config or copy.deepcopy(DEFAULT_ACTION_GEN_CONFIG)
        self.conservative_score_config = conservative_score_config or copy.deepcopy(DEFAULT_CONSERVATIVE_SCORE_CONFIG)
        self.exploration_config = exploration_config or copy.deepcopy(DEFAULT_EXPLORATION_CONFIG)
        self.reward_config = reward_config or copy.deepcopy(DEFAULT_REWARD_CONFIG)

        # Q-values keyed by (state_key, action)
        self.q: Dict[Tuple[Tuple, Action], float] = defaultdict(float)
        # Visit counts for diagnostics and exploration encouragement
        self.state_visits: Dict[Tuple, int] = defaultdict(int)
        self.state_action_visits: Dict[Tuple[Tuple, Action], int] = defaultdict(int)

    def epsilon_for_episode(self, episode_index: int) -> float:
        """Episode index is 1-based."""
        if episode_index <= self.explore_iterations:
            # Pure exploration phase: seek broad state coverage
            return 1.0
        # Post-exploration: settle on conservative/best learned path
        steps = episode_index - self.explore_iterations
        eps = self.epsilon_decay ** steps
        return max(self.min_epsilon, eps)

    def _conservative_score(self, action: Action, obs: Observation) -> float:
        """Heuristic score favoring survival-oriented, conservative choices."""
        cfg = self.conservative_score_config
        buy, sell, feed, plant = action
        needed = obs.population * HamurabiGame.BUSHELS_PER_PERSON
        score = 0.0
        # Strongly prefer feeding everyone (or as much as possible)
        if needed <= 0:
            feed_ratio = 1.0
        else:
            feed_ratio = min(1.0, feed / needed)
        score += cfg.feed_weight * feed_ratio
        if feed >= needed:
            score += cfg.full_feed_bonus
        # Prefer planting a reasonable fraction of feasible capacity
        max_farmable = min(obs.acres + buy - sell, obs.population * HamurabiGame.ACRES_PER_PERSON)
        if max_farmable > 0:
            score += cfg.plant_weight * (plant / max_farmable)
        # Mild preference to avoid reckless large land sales when population is stable
        if sell > obs.acres * cfg.sell_half_threshold and obs.population > 0:
            score -= cfg.sell_half_penalty
        # Small penalty for buying very expensive land with low reserves
        if buy > 0 and obs.land_price >= cfg.buy_expensive_price and obs.bushels < needed * cfg.buy_expensive_reserve_mult:
            score -= cfg.buy_expensive_penalty
        return score

    def _best_action(self, state_key: Tuple, actions: Sequence[Action], obs: Observation) -> Action:
        # Break ties using conservative heuristic to settle on a safe/best path
        best_val = None
        best_actions: List[Action] = []
        for a in actions:
            val = self.q[(state_key, a)]
            if best_val is None or val > best_val + 1e-12:
                best_val = val
                best_actions = [a]
            elif abs(val - best_val) <= 1e-12:
                best_actions.append(a)
        if len(best_actions) == 1:
            return best_actions[0]
        # Among Q-ties (common for unseen states), choose most conservative
        return max(best_actions, key=lambda a: (self._conservative_score(a, obs), -self.state_action_visits[(state_key, a)]))

    def select_action(
        self,
        game: HamurabiGame,
        obs: Observation,
        episode_index: int,
        explore_bias: bool = True,
    ) -> Tuple[Action, Tuple]:
        """Select an action for the given observation.

        During the exploration phase (first N iterations), selection is random
        but biased toward viable, less-visited actions so the agent explores
        broadly across many game states (including later years) rather than
        ending episodes immediately.
        """
        exp_cfg = self.exploration_config
        # Use SimpleAI SearchProblem abstraction to obtain candidate actions
        problem = HamurabiTurnProblem(game, obs, self.action_gen_config)
        actions = problem.actions(obs)
        if not actions:
            actions = [(0, 0, 0, 0)]

        state_key = discretize_state(obs, self.discretization_config)
        self.state_visits[state_key] += 1
        epsilon = self.epsilon_for_episode(episode_index)

        if self.rng.random() < epsilon:
            if episode_index <= self.explore_iterations and explore_bias:
                # Seek broad state coverage: prefer actions that can sustain life
                # (so episodes reach varied later states) while favoring
                # less-tried options to diversify visited states.
                needed = obs.population * HamurabiGame.BUSHELS_PER_PERSON
                viable = [a for a in actions if a[2] >= int(exp_cfg.viable_feed_ratio * needed)] or [
                    a for a in actions if a[2] >= int(exp_cfg.viable_fallback_ratio * needed)
                ] or actions
                # Mix: mostly viable for reachability, sometimes fully random
                pool = viable if self.rng.random() < exp_cfg.viable_pool_prob else actions
                counts = [self.state_action_visits[(state_key, a)] for a in pool]
                min_c = min(counts)
                least = [a for a, c in zip(pool, counts) if c == min_c]
                # Break ties by a small random jitter over heuristic to vary paths
                action = max(
                    least,
                    key=lambda a: self._conservative_score(a, obs) + self.rng.random() * exp_cfg.heuristic_jitter,
                )
            else:
                action = self.rng.choice(actions)
        else:
            action = self._best_action(state_key, actions, obs)

        self.state_action_visits[(state_key, action)] += 1
        return action, state_key

    def update(
        self,
        state_key: Tuple,
        action: Action,
        reward: float,
        next_state_key: Optional[Tuple],
        next_actions: Optional[Sequence[Action]],
        done: bool,
    ) -> None:
        current = self.q[(state_key, action)]
        if done or not next_state_key or not next_actions:
            target = reward
        else:
            best_next = max(self.q[(next_state_key, a)] for a in next_actions)
            target = reward + self.gamma * best_next
        self.q[(state_key, action)] = current + self.alpha * (target - current)


def compute_reward(
    prev: GameState, new: GameState, done: bool, config: Optional[RewardConfig] = None
) -> float:
    """Reward shaping to encourage survival and conservative, effective rule.

    - Heavy penalties for impeachment or population wipeout.
    - Penalties for starvation; strong rewards for feeding everyone.
    - Bonus for completing the full term in good standing.
    """
    if config is None:
        config = DEFAULT_REWARD_CONFIG

    reward = 0.0

    # Survival baseline
    if new.impeached:
        reward -= config.impeach_penalty
    elif done and new.population <= 0:
        reward -= config.wipeout_penalty
    else:
        reward += config.survival_bonus

    # Starvation penalties / feeding rewards
    if prev.population > 0:
        starved_ratio = new.starved / prev.population
        reward -= config.starve_penalty_scale * starved_ratio
        if new.starved == 0:
            reward += config.no_starve_bonus
        elif starved_ratio <= config.low_starve_ratio_threshold:
            reward += config.low_starve_bonus

    # Encourage sustainable reserves and production capacity
    if not (new.impeached or new.population <= 0):
        reserves_per_person = new.bushels / max(1, new.population)
        if reserves_per_person < config.reserves_low_threshold:
            reward -= config.reserves_low_penalty
        elif reserves_per_person >= config.reserves_high_threshold:
            reward += config.reserves_high_bonus
        acres_per_person = new.acres / max(1, new.population)
        if config.app_good_low <= acres_per_person <= config.app_good_high:
            reward += config.app_good_bonus
        elif acres_per_person < config.app_low_threshold:
            reward -= config.app_low_penalty

    # Completion bonus (conservative, successful term)
    if done and not new.impeached and new.population > 0 and new.year > HamurabiGame.MAX_YEARS:
        acres_per_person = new.acres / new.population
        avg_starv = new.cumulative_starvation_pct / max(1, new.years_played)
        if avg_starv <= config.completion_tier1_avg_starv and acres_per_person >= config.completion_tier1_app:
            reward += config.completion_tier1_bonus
        elif avg_starv <= config.completion_tier2_avg_starv and acres_per_person >= config.completion_tier2_app:
            reward += config.completion_tier2_bonus
        else:
            reward += config.completion_tier3_bonus

    return reward


@dataclass
class EpisodeResult:
    episode: int
    survived: bool
    completed_term: bool
    impeached: bool
    years_played: int
    final_population: int
    final_acres: int
    final_bushels: int
    total_starved: int
    total_reward: float
    unique_states_seen: int


def run_episode(
    agent: QLearningHamurabiAgent,
    episode_index: int,
    training: bool = True,
    seed: Optional[int] = None,
) -> Tuple[EpisodeResult, set]:
    """Run a single Hamurabi episode with the agent."""
    rng = random.Random(seed)
    game = HamurabiGame(rng=rng)
    total_reward = 0.0
    visited_states = set()

    while not game.state.game_over:
        land_price = game.current_land_price()
        s = game.state
        obs = Observation(
            year=s.year,
            population=s.population,
            bushels=s.bushels,
            acres=s.acres,
            land_price=land_price,
        )
        state_key = discretize_state(obs, agent.discretization_config)
        visited_states.add(state_key)

        action, chosen_state_key = agent.select_action(game, obs, episode_index)
        # chosen_state_key should match state_key; use the discretized key
        prev_snapshot = GameState(**vars(s))

        ok, _ = game.run_year(*action, land_price=land_price)
        if not ok:
            # Should not happen because actions come from validated candidates,
            # but fall back safely by applying a no-op-like valid action.
            fallback = (0, 0, min(s.bushels, s.population * HamurabiGame.BUSHELS_PER_PERSON), 0)
            ok2, _ = game.validate_turn(*fallback, land_price)
            if ok2:
                game.advance_year(*fallback, land_price)
                action = fallback
            else:
                # As a last resort, force-feed/plant nothing validly
                game.advance_year(0, 0, 0, 0, land_price)

        new_s = game.state
        done = new_s.game_over
        reward = compute_reward(prev_snapshot, new_s, done, agent.reward_config)
        total_reward += reward

        if done:
            next_state_key = None
            next_actions = None
        else:
            # For bootstrapping, derive next state's candidate actions using a
            # temporary game-like view? We need next land price context; but
            # land price is determined at start of each year. Approximate by
            # using a representative price for action-value expectation, or
            # simply compute candidates once next year's price is known in the
            # next iteration. To keep Q-update consistent with on-policy next
            # step, we defer bootstrapping to next iteration's selection; here
            # use a neutral next value by sampling a price for candidate set.
            next_price = rng.randint(17, 26)
            next_obs = Observation(
                year=new_s.year,
                population=new_s.population,
                bushels=new_s.bushels,
                acres=new_s.acres,
                land_price=next_price,
            )
            next_state_key = discretize_state(next_obs, agent.discretization_config)
            # Build candidate actions for next state with a temp game snapshot
            temp_game = HamurabiGame(rng=random.Random(rng.random()))
            temp_game.state = GameState(**vars(new_s))
            problem = HamurabiTurnProblem(temp_game, next_obs, agent.action_gen_config)
            next_actions = problem.actions(next_obs)

        if training:
            agent.update(chosen_state_key, action, reward, next_state_key, next_actions, done)

    st = game.state
    completed = st.game_over and not st.impeached and st.population > 0 and st.years_played >= HamurabiGame.MAX_YEARS
    survived = not st.impeached and st.population > 0
    result = EpisodeResult(
        episode=episode_index,
        survived=survived,
        completed_term=completed,
        impeached=st.impeached,
        years_played=st.years_played,
        final_population=st.population,
        final_acres=st.acres,
        final_bushels=st.bushels,
        total_starved=st.total_starved,
        total_reward=total_reward,
        unique_states_seen=len(visited_states),
    )
    return result, visited_states


def train(
    episodes: int = 300,
    explore_iterations: int = 100,
    seed: int = 42,
    verbose: bool = True,
    alpha: float = 0.25,
    gamma: float = 0.98,
    min_epsilon: float = 0.02,
    epsilon_decay: float = 0.95,
    discretization_config: Optional[DiscretizationConfig] = None,
    action_gen_config: Optional[ActionGenConfig] = None,
    conservative_score_config: Optional[ConservativeScoreConfig] = None,
    exploration_config: Optional[ExplorationConfig] = None,
    reward_config: Optional[RewardConfig] = None,
) -> Tuple[QLearningHamurabiAgent, List[EpisodeResult], dict]:
    """Train the Q-learning agent over a number of episodes."""
    rng = random.Random(seed)
    agent = QLearningHamurabiAgent(
        alpha=alpha,
        gamma=gamma,
        explore_iterations=explore_iterations,
        min_epsilon=min_epsilon,
        epsilon_decay=epsilon_decay,
        seed=rng.randint(0, 10**9),
        discretization_config=discretization_config,
        action_gen_config=action_gen_config,
        conservative_score_config=conservative_score_config,
        exploration_config=exploration_config,
        reward_config=reward_config,
    )

    results: List[EpisodeResult] = []
    all_states_seen = set()
    explore_phase_states = set()
    exploit_phase_states = set()

    for ep in range(1, episodes + 1):
        ep_seed = rng.randint(0, 10**9)
        res, visited = run_episode(agent, ep, training=True, seed=ep_seed)
        results.append(res)
        all_states_seen.update(visited)
        if ep <= explore_iterations:
            explore_phase_states.update(visited)
        else:
            exploit_phase_states.update(visited)

        if verbose and (ep == 1 or ep == explore_iterations or ep % 50 == 0 or ep == episodes):
            window = results[max(0, ep - 20):ep]
            avg_reward = sum(r.total_reward for r in window) / len(window)
            completions = sum(1 for r in window if r.completed_term)
            eps = agent.epsilon_for_episode(ep)
            phase = "EXPLORE" if ep <= explore_iterations else "EXPLOIT"
            print(
                f"Episode {ep:4d} [{phase}] eps={eps:.3f} "
                f"avg_reward(last {len(window)})={avg_reward:7.2f} "
                f"completed(last {len(window)})={completions:2d} "
                f"states_seen_total={len(all_states_seen)}"
            )

    stats = {
        "episodes": episodes,
        "explore_iterations": explore_iterations,
        "unique_states_total": len(all_states_seen),
        "unique_states_explore_phase": len(explore_phase_states),
        "unique_states_exploit_phase": len(exploit_phase_states),
        "q_table_entries": len(agent.q),
    }
    return agent, results, stats


def evaluate_policy(
    agent: QLearningHamurabiAgent,
    episodes: int = 50,
    seed: int = 123,
    verbose: bool = False,
) -> List[EpisodeResult]:
    """Evaluate the learned (greedy) policy without exploration."""
    rng = random.Random(seed)
    # Force greedy behavior by evaluating as if far past exploration phase
    eval_episode_index = agent.explore_iterations + 10_000
    results: List[EpisodeResult] = []
    all_states = set()

    for i in range(1, episodes + 1):
        ep_seed = rng.randint(0, 10**9)
        res, visited = run_episode(agent, eval_episode_index, training=False, seed=ep_seed)
        results.append(res)
        all_states.update(visited)
        if verbose:
            status = "COMPLETED" if res.completed_term else ("IMPEACHED" if res.impeached else "FAILED")
            print(
                f"Eval {i:2d}: {status} years={res.years_played} pop={res.final_population} "
                f"acres={res.final_acres} starved={res.total_starved} reward={res.total_reward:.1f}"
            )

    if verbose or episodes > 0:
        completed = sum(1 for r in results if r.completed_term)
        impeached = sum(1 for r in results if r.impeached)
        avg_reward = sum(r.total_reward for r in results) / max(1, len(results))
        avg_pop = sum(r.final_population for r in results) / max(1, len(results))
        print(
            f"Evaluation summary: {completed}/{episodes} completed term, "
            f"{impeached}/{episodes} impeached, avg_reward={avg_reward:.2f}, avg_final_pop={avg_pop:.1f}"
        )
    return results


def play_greedy_game(agent: QLearningHamurabiAgent, seed: int = 7, verbose: bool = True) -> EpisodeResult:
    """Play one game using the learned greedy policy and optionally print steps."""
    rng = random.Random(seed)
    game = HamurabiGame(rng=rng)
    eval_episode_index = agent.explore_iterations + 10_000
    total_reward = 0.0
    visited = set()

    if verbose:
        print("\n--- Greedy policy demonstration game ---")

    while not game.state.game_over:
        land_price = game.current_land_price()
        s = game.state
        obs = Observation(
            year=s.year, population=s.population, bushels=s.bushels, acres=s.acres, land_price=land_price
        )
        state_key = discretize_state(obs, agent.discretization_config)
        visited.add(state_key)
        action, chosen_key = agent.select_action(game, obs, eval_episode_index)
        buy, sell, feed, plant = action
        prev = GameState(**vars(s))
        ok, err = game.run_year(*action, land_price=land_price)
        if not ok:
            fallback = (0, 0, min(s.bushels, s.population * HamurabiGame.BUSHELS_PER_PERSON), 0)
            game.advance_year(*fallback, land_price)
            action = fallback
            buy, sell, feed, plant = action

        new_s = game.state
        reward = compute_reward(prev, new_s, new_s.game_over, agent.reward_config)
        total_reward += reward

        if verbose:
            trade = f"bought {buy}" if buy else (f"sold {sell}" if sell else "no trade")
            print(
                f"Year {obs.year}: price={land_price}, pop={obs.population}, bushels={obs.bushels}, acres={obs.acres} -> "
                f"{trade}, fed={feed}, planted={plant} | "
                f"starved={new_s.starved}, pop_now={new_s.population}, bushels_now={new_s.bushels}, acres_now={new_s.acres}"
            )
            if new_s.plague:
                print("  (plague struck)")

    st = game.state
    completed = st.game_over and not st.impeached and st.population > 0 and st.years_played >= HamurabiGame.MAX_YEARS
    survived = not st.impeached and st.population > 0
    result = EpisodeResult(
        episode=0,
        survived=survived,
        completed_term=completed,
        impeached=st.impeached,
        years_played=st.years_played,
        final_population=st.population,
        final_acres=st.acres,
        final_bushels=st.bushels,
        total_starved=st.total_starved,
        total_reward=total_reward,
        unique_states_seen=len(visited),
    )
    if verbose:
        print(f"\nOutcome: {'Completed term' if completed else ('Impeached' if st.impeached else 'Failed')}")
        print(f"Years played: {st.years_played}, final pop: {st.population}, final acres: {st.acres}, total starved: {st.total_starved}")
        if st.end_message:
            print("Assessment:")
            print(st.end_message)
    return result


def _print_phase_summary(results: List[EpisodeResult], explore_iterations: int) -> None:
    explore = results[:explore_iterations]
    exploit = results[explore_iterations:]
    def summarize(label: str, chunk: List[EpisodeResult]):
        if not chunk:
            print(f"{label}: no episodes")
            return
        completed = sum(1 for r in chunk if r.completed_term)
        impeached = sum(1 for r in chunk if r.impeached)
        avg_reward = sum(r.total_reward for r in chunk) / len(chunk)
        avg_states = sum(r.unique_states_seen for r in chunk) / len(chunk)
        print(
            f"{label}: n={len(chunk)}, completed={completed} ({completed/len(chunk)*100:.1f}%), "
            f"impeached={impeached} ({impeached/len(chunk)*100:.1f}%), "
            f"avg_reward={avg_reward:.2f}, avg_unique_states_per_episode={avg_states:.2f}"
        )
    print("\nPhase summary:")
    summarize(f"Exploration (first {explore_iterations})", explore)
    summarize("Post-exploration (conservative/exploit)", exploit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a reinforcement-learning Hamurabi agent (using SimpleAI).")
    parser.add_argument("--episodes", type=int, default=300, help="Total training episodes/iterations (default: 300)")
    parser.add_argument("--explore-iterations", type=int, default=100, help="Number of initial exploratory iterations (default: 100)")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Greedy evaluation episodes after training (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training (default: 42)")
    parser.add_argument("--demo-seed", type=int, default=7, help="Seed for demo greedy game (default: 7)")
    parser.add_argument("--quiet", action="store_true", help="Reduce training output")
    args = parser.parse_args()

    print("Hamurabi RL Agent")
    print(" - Reinforcement learning: tabular Q-learning interacting with hamurabi.HamurabiGame")
    print(" - SimpleAI: uses simpleai.search.models.SearchProblem (HamurabiTurnProblem) for action modeling")
    print(f" - Training: {args.episodes} episodes, first {args.explore_iterations} exploratory (epsilon=1.0), then conservative decay")
    print()

    agent, results, stats = train(
        episodes=args.episodes,
        explore_iterations=args.explore_iterations,
        seed=args.seed,
        verbose=not args.quiet,
    )

    print("\nTraining coverage stats:")
    print(f" - Unique discretized states (total): {stats['unique_states_total']}")
    print(f" - Unique discretized states (exploration phase): {stats['unique_states_explore_phase']}")
    print(f" - Unique discretized states (post-exploration phase): {stats['unique_states_exploit_phase']}")
    print(f" - Q-table entries: {stats['q_table_entries']}")
    _print_phase_summary(results, args.explore_iterations)

    print("\nGreedy policy evaluation (no exploration):")
    evaluate_policy(agent, episodes=args.eval_episodes, seed=12345, verbose=False)

    play_greedy_game(agent, seed=args.demo_seed, verbose=True)


if __name__ == "__main__":
    main()
