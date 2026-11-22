#!/usr/bin/env python3
"""Evaluation metrics and heuristic agents for Hamurabi.

Provides:
- Standardized evaluation metrics over repeated game runs.
- Several rule-based heuristic policies ("agents") to compare.
- Utilities to evaluate RL agents with different training configurations.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from hamurabi import HamurabiGame, GameState
import hamurabi_rl_agent as rl


Action = Tuple[int, int, int, int]  # (buy, sell, feed, plant)


@dataclass
class GameMetrics:
    completed_term: bool
    survived: bool
    impeached: bool
    years_played: int
    final_population: int
    final_acres: int
    final_bushels: int
    acres_per_person: float
    total_starved: int
    avg_starvation_pct: float
    total_reward: float


@dataclass
class AggregateMetrics:
    name: str
    games: int
    completion_rate: float
    survival_rate: float
    impeachment_rate: float
    avg_years_played: float
    avg_final_population: float
    avg_final_acres: float
    avg_acres_per_person: float
    avg_total_starved: float
    avg_starvation_pct: float
    avg_total_reward: float
    median_total_reward: float
    score: float


def _run_policy_episode(
    policy: Callable[[HamurabiGame, int], Action],
    seed: int,
) -> GameMetrics:
    rng = random.Random(seed)
    game = HamurabiGame(rng=rng)
    total_reward = 0.0

    while not game.state.game_over:
        land_price = game.current_land_price()
        s = game.state
        prev = GameState(**vars(s))
        action = policy(game, land_price)
        # Ensure action is valid; if not, try conservative fallback
        ok, _ = game.validate_turn(*action, land_price)
        if not ok:
            action = conservative_fallback_action(game, land_price)
            ok, _ = game.validate_turn(*action, land_price)
            if not ok:
                action = (0, 0, min(s.bushels, s.population * HamurabiGame.BUSHELS_PER_PERSON), 0)
        game.advance_year(*action, land_price)
        total_reward += rl.compute_reward(prev, game.state, game.state.game_over)

    st = game.state
    completed = st.game_over and not st.impeached and st.population > 0 and st.years_played >= HamurabiGame.MAX_YEARS
    survived = not st.impeached and st.population > 0
    acres_per_person = st.acres / st.population if st.population > 0 else 0.0
    avg_starv = st.cumulative_starvation_pct / max(1, st.years_played)
    return GameMetrics(
        completed_term=completed,
        survived=survived,
        impeached=st.impeached,
        years_played=st.years_played,
        final_population=st.population,
        final_acres=st.acres,
        final_bushels=st.bushels,
        acres_per_person=acres_per_person,
        total_starved=st.total_starved,
        avg_starvation_pct=avg_starv,
        total_reward=total_reward,
    )


def aggregate_metrics(name: str, results: Sequence[GameMetrics]) -> AggregateMetrics:
    n = len(results)
    if n == 0:
        raise ValueError("no results")
    completion_rate = sum(1 for r in results if r.completed_term) / n
    survival_rate = sum(1 for r in results if r.survived) / n
    impeachment_rate = sum(1 for r in results if r.impeached) / n
    avg_years = sum(r.years_played for r in results) / n
    avg_pop = sum(r.final_population for r in results) / n
    avg_acres = sum(r.final_acres for r in results) / n
    avg_app = sum(r.acres_per_person for r in results) / n
    avg_starved = sum(r.total_starved for r in results) / n
    avg_starv_pct = sum(r.avg_starvation_pct for r in results) / n
    rewards = [r.total_reward for r in results]
    avg_reward = sum(rewards) / n
    med_reward = statistics.median(rewards)

    # Composite score emphasizing reliable completion and low starvation.
    # Scaled to be readable; higher is better.
    score = (
        100.0 * completion_rate
        + 10.0 * survival_rate
        - 80.0 * impeachment_rate
        + 0.2 * avg_pop
        + 0.05 * avg_acres
        + 2.0 * avg_app
        - 0.5 * avg_starved
        - 1.0 * avg_starv_pct
        + 0.1 * avg_years
    )
    return AggregateMetrics(
        name=name,
        games=n,
        completion_rate=completion_rate,
        survival_rate=survival_rate,
        impeachment_rate=impeachment_rate,
        avg_years_played=avg_years,
        avg_final_population=avg_pop,
        avg_final_acres=avg_acres,
        avg_acres_per_person=avg_app,
        avg_total_starved=avg_starved,
        avg_starvation_pct=avg_starv_pct,
        avg_total_reward=avg_reward,
        median_total_reward=med_reward,
        score=score,
    )


def evaluate_policy_fn(
    name: str,
    policy: Callable[[HamurabiGame, int], Action],
    games: int = 200,
    seed: int = 12345,
) -> AggregateMetrics:
    rng = random.Random(seed)
    results = [_run_policy_episode(policy, rng.randint(0, 10**9)) for _ in range(games)]
    return aggregate_metrics(name, results)


def conservative_fallback_action(game: HamurabiGame, land_price: int) -> Action:
    """Safe fallback: feed everyone possible, plant as much as feasible, no trade."""
    s = game.state
    feed = min(s.bushels, s.population * HamurabiGame.BUSHELS_PER_PERSON)
    bushels_after_feed = s.bushels - feed
    max_by_seed = bushels_after_feed * 2
    farm_cap = min(s.acres, s.population * HamurabiGame.ACRES_PER_PERSON)
    plant = max(0, min(farm_cap, max_by_seed))
    action = (0, 0, feed, plant)
    ok, _ = game.validate_turn(*action, land_price)
    if ok:
        return action
    # degrade gracefully
    return (0, 0, min(s.bushels, s.population * HamurabiGame.BUSHELS_PER_PERSON), 0)


# -------------------- Heuristic policies --------------------

def heuristic_feed_all_plant_max(game: HamurabiGame, land_price: int) -> Action:
    """Conservative baseline: feed all people, plant max feasible, no trading."""
    s = game.state
    feed = min(s.bushels, s.population * HamurabiGame.BUSHELS_PER_PERSON)
    remaining = s.bushels - feed
    plant = min(s.acres, s.population * HamurabiGame.ACRES_PER_PERSON, remaining * 2)
    return (0, 0, feed, max(0, plant))


def heuristic_feed_all_plant_conservative(game: HamurabiGame, land_price: int) -> Action:
    """Feed all, plant conservatively keeping extra grain buffer."""
    s = game.state
    feed = min(s.bushels, s.population * HamurabiGame.BUSHELS_PER_PERSON)
    remaining = s.bushels - feed
    # Keep ~1 year buffer for population if possible
    buffer = min(remaining, s.population * HamurabiGame.BUSHELS_PER_PERSON)
    plant_budget = max(0, remaining - buffer)
    plant = min(s.acres, s.population * HamurabiGame.ACRES_PER_PERSON, plant_budget * 2)
    if plant <= 0:
        plant = min(s.acres, s.population * HamurabiGame.ACRES_PER_PERSON, remaining * 2)
    return (0, 0, feed, max(0, plant))


def heuristic_feed_all_trade_for_planting(game: HamurabiGame, land_price: int) -> Action:
    """Feed all, then buy/sell land to roughly match plantable capacity & people."""
    s = game.state
    acres = s.acres
    bushels = s.bushels
    buy = sell = 0
    # Target acres around what population can farm (10 per person), but keep grain for feeding+seed.
    desired_acres = s.population * HamurabiGame.ACRES_PER_PERSON
    if land_price <= 20 and acres < desired_acres:
        # try to buy a bit if affordable while keeping enough to feed and seed
        need_feed = s.population * HamurabiGame.BUSHELS_PER_PERSON
        # estimate seed for desired acres
        can_spend = bushels - need_feed - (desired_acres // 2) - 200
        if can_spend > 0:
            buy = min((desired_acres - acres), can_spend // land_price)
            buy = max(0, buy)
    elif land_price >= 24 and acres > desired_acres:
        sell = min(acres - desired_acres, acres // 4)
        sell = max(0, sell)

    # simulate trade
    bushels_after = bushels - buy * land_price + sell * land_price
    acres_after = acres + buy - sell
    feed = min(bushels_after, s.population * HamurabiGame.BUSHELS_PER_PERSON)
    remaining = bushels_after - feed
    plant = min(acres_after, s.population * HamurabiGame.ACRES_PER_PERSON, remaining * 2)
    return (max(0, buy), max(0, sell), feed, max(0, plant))


def heuristic_balanced_reserve(game: HamurabiGame, land_price: int) -> Action:
    """Feed all, keep a reserve, plant high but not grain-starving; opportunistic trade."""
    s = game.state
    acres = s.acres
    bushels = s.bushels
    buy = sell = 0
    desired_acres = int(s.population * 10)

    # Mild opportunistic trading
    if land_price <= 18 and acres < desired_acres + 50:
        need_feed = s.population * 20
        reserve = need_feed
        affordable = max(0, (bushels - need_feed - reserve - desired_acres // 2) // land_price)
        buy = min(affordable, max(0, desired_acres - acres + 20), 60)
    elif land_price >= 25 and acres > desired_acres + 100:
        sell = min(acres - desired_acres, 80)

    bushels_after = bushels - buy * land_price + sell * land_price
    acres_after = acres + buy - sell
    feed = min(bushels_after, s.population * 20)
    remaining = bushels_after - feed
    reserve = min(remaining, int(s.population * 20 * 0.5))
    plant_budget = max(0, remaining - reserve)
    plant = min(acres_after, s.population * 10, plant_budget * 2)
    if plant <= 0 and remaining > 0:
        plant = min(acres_after, s.population * 10, remaining * 2)
    return (max(0, buy), max(0, sell), feed, max(0, plant))


def heuristic_aggressive_expansion(game: HamurabiGame, land_price: int) -> Action:
    """Try to buy land aggressively and plant a lot; still feed everyone first."""
    s = game.state
    acres = s.acres
    bushels = s.bushels
    feed = min(bushels, s.population * 20)
    remaining = bushels - feed
    buy = 0
    if land_price <= 22 and remaining > s.population * 10:
        spend = max(0, remaining - s.population * 15)
        buy = spend // land_price
    bushels_after = bushels - buy * land_price
    acres_after = acres + buy
    feed = min(bushels_after, s.population * 20)
    remaining = bushels_after - feed
    plant = min(acres_after, s.population * 10, remaining * 2)
    return (max(0, buy), 0, feed, max(0, plant))


def heuristic_sell_surplus_land(game: HamurabiGame, land_price: int) -> Action:
    """Sell land beyond farming capacity to build reserves, feed all, plant max."""
    s = game.state
    acres = s.acres
    bushels = s.bushels
    farm_cap = s.population * 10
    sell = max(0, acres - farm_cap) if land_price >= 20 else max(0, acres - farm_cap - 50)
    sell = min(sell, acres)
    bushels_after = bushels + sell * land_price
    acres_after = acres - sell
    feed = min(bushels_after, s.population * 20)
    remaining = bushels_after - feed
    plant = min(acres_after, s.population * 10, remaining * 2)
    return (0, sell, feed, max(0, plant))


HEURISTICS: Dict[str, Callable[[HamurabiGame, int], Action]] = {
    "feed_all_plant_max": heuristic_feed_all_plant_max,
    "feed_all_plant_conservative_buffer": heuristic_feed_all_plant_conservative,
    "feed_all_trade_for_planting": heuristic_feed_all_trade_for_planting,
    "balanced_reserve_opportunistic": heuristic_balanced_reserve,
    "aggressive_expansion": heuristic_aggressive_expansion,
    "sell_surplus_land": heuristic_sell_surplus_land,
}


# -------------------- RL evaluation helpers --------------------

def episode_result_to_metrics(r: rl.EpisodeResult) -> GameMetrics:
    # EpisodeResult from RL module doesn't include acres_per_person/avg_starvation_pct
    # directly in the same form, so derive from available fields.
    acres_per_person = r.final_acres / r.final_population if r.final_population > 0 else 0.0
    # avg starvation pct per year not stored; approximate unavailable -> use 0 and rely on totals.
    # Better: recompute not possible without game state trace; store as NaN-like via 0? keep 0.
    return GameMetrics(
        completed_term=r.completed_term,
        survived=r.survived,
        impeached=r.impeached,
        years_played=r.years_played,
        final_population=r.final_population,
        final_acres=r.final_acres,
        final_bushels=r.final_bushels,
        acres_per_person=acres_per_person,
        total_starved=r.total_starved,
        avg_starvation_pct=0.0,
        total_reward=r.total_reward,
    )


def evaluate_trained_agent(
    agent: rl.QLearningHamurabiAgent,
    games: int = 100,
    seed: int = 12345,
) -> List[GameMetrics]:
    results = rl.evaluate_policy(agent, episodes=games, seed=seed, verbose=False)
    return [episode_result_to_metrics(r) for r in results]


@dataclass
class RLExperimentConfig:
    name: str
    episodes: int
    explore_iterations: int
    alpha: float = 0.25
    gamma: float = 0.98
    min_epsilon: float = 0.02
    epsilon_decay: float = 0.95


@dataclass
class RLExperimentResult:
    config: RLExperimentConfig
    aggregate: AggregateMetrics
    per_seed: List[AggregateMetrics]
    avg_train_states_total: float
    avg_train_states_explore: float
    avg_q_table_entries: float


def run_rl_experiment(
    config: RLExperimentConfig,
    train_seeds: Sequence[int],
    eval_games: int = 100,
    eval_seed_base: int = 50000,
) -> RLExperimentResult:
    per_seed_aggs: List[AggregateMetrics] = []
    all_eval_metrics: List[GameMetrics] = []
    states_total = []
    states_explore = []
    q_entries = []

    for i, ts in enumerate(train_seeds):
        agent, _train_results, stats = rl.train(
            episodes=config.episodes,
            explore_iterations=config.explore_iterations,
            seed=ts,
            verbose=False,
            alpha=config.alpha,
            gamma=config.gamma,
            min_epsilon=config.min_epsilon,
            epsilon_decay=config.epsilon_decay,
        )
        eval_metrics = evaluate_trained_agent(agent, games=eval_games, seed=eval_seed_base + i * 1000)
        per_seed_aggs.append(aggregate_metrics(f"{config.name}_seed{ts}", eval_metrics))
        all_eval_metrics.extend(eval_metrics)
        states_total.append(stats.get("unique_states_total", 0))
        states_explore.append(stats.get("unique_states_explore_phase", 0))
        q_entries.append(stats.get("q_table_entries", 0))

    agg = aggregate_metrics(config.name, all_eval_metrics)
    return RLExperimentResult(
        config=config,
        aggregate=agg,
        per_seed=per_seed_aggs,
        avg_train_states_total=sum(states_total) / len(states_total) if states_total else 0.0,
        avg_train_states_explore=sum(states_explore) / len(states_explore) if states_explore else 0.0,
        avg_q_table_entries=sum(q_entries) / len(q_entries) if q_entries else 0.0,
    )


def format_aggregate(m: AggregateMetrics) -> str:
    return (
        f"{m.name}: games={m.games}, completion={m.completion_rate*100:.1f}%, "
        f"survival={m.survival_rate*100:.1f}%, impeachment={m.impeachment_rate*100:.1f}%, "
        f"avg_years={m.avg_years_played:.2f}, avg_pop={m.avg_final_population:.1f}, "
        f"avg_acres={m.avg_final_acres:.1f}, avg_acres/person={m.avg_acres_per_person:.2f}, "
        f"avg_starved={m.avg_total_starved:.1f}, avg_reward={m.avg_total_reward:.2f}, "
        f"median_reward={m.median_total_reward:.2f}, score={m.score:.2f}"
    )


def format_rl_result(r: RLExperimentResult) -> str:
    c = r.config
    base = format_aggregate(r.aggregate)
    return (
        f"[RL {c.name}] episodes={c.episodes}, explore={c.explore_iterations}, "
        f"eps_decay={c.epsilon_decay}, min_eps={c.min_epsilon}, alpha={c.alpha}, gamma={c.gamma} | "
        f"train_states(total/explore)={r.avg_train_states_total:.1f}/{r.avg_train_states_explore:.1f}, "
        f"avg_q={r.avg_q_table_entries:.0f} || {base}"
    )
