#!/usr/bin/env python3
"""Genetic algorithm agent for Hamurabi.

Evolves a parameterized policy (genome) that interacts with the Hamurabi game
implementation in ``hamurabi.py``. Uses SimpleAI's local-search genetic
algorithm (``simpleai.search.local.genetic``) to optimize policy parameters for
high fitness across simulated games.

Genome (policy parameters):
  0. feed_mult               - feed target as multiple of required grain (pop*20)
  1. reserve_per_person      - bushels to reserve per person after feeding
  2. plant_fraction          - fraction of feasible planting capacity to plant
  3. buy_if_price_le         - buy land when price <= this threshold
  4. buy_target_app          - target acres per person when buying
  5. sell_if_price_ge        - sell land when price >= this threshold
  6. sell_keep_app           - keep at least this many acres per person when selling
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from simpleai.search.local import genetic
from simpleai.search.models import SearchProblem

from hamurabi import HamurabiGame, GameState


Action = Tuple[int, int, int, int]  # (buy, sell, feed, plant)
Genome = Tuple[float, ...]


# (min, max) bounds for each gene
GENE_BOUNDS: Tuple[Tuple[float, float], ...] = (
    (0.90, 1.30),  # feed_mult
    (0.0, 30.0),   # reserve_per_person
    (0.50, 1.00),  # plant_fraction
    (17.0, 26.0),  # buy_if_price_le
    (7.0, 14.0),   # buy_target_app
    (17.0, 26.0),  # sell_if_price_ge
    (7.0, 14.0),   # sell_keep_app
)


def clamp_genome(genome: Sequence[float]) -> Genome:
    clamped = []
    for v, (lo, hi) in zip(genome, GENE_BOUNDS):
        clamped.append(max(lo, min(hi, float(v))))
    return tuple(clamped)


def random_genome(rng: random.Random) -> Genome:
    return tuple(rng.uniform(lo, hi) for lo, hi in GENE_BOUNDS)


def decode_action(genome: Genome, game: HamurabiGame, land_price: int) -> Action:
    """Decode a genome into a valid action for the current game state."""
    feed_mult, reserve_pp, plant_frac, buy_if_le, buy_target_app, sell_if_ge, sell_keep_app = genome
    s = game.state
    pop = max(1, s.population)
    acres = s.acres
    bushels = s.bushels

    buy = 0
    sell = 0

    # Trading decision (cannot both buy and sell)
    if land_price <= buy_if_le:
        target_acres = int(buy_target_app * pop)
        desired_buy = max(0, target_acres - acres)
        # Keep enough grain to feed fully and hold a small reserve/seed margin
        needed_feed = int(feed_mult * pop * HamurabiGame.BUSHELS_PER_PERSON)
        reserve = int(reserve_pp * pop)
        max_spend = max(0, bushels - needed_feed - reserve - (target_acres // 4))
        affordable = max_spend // max(1, land_price)
        buy = min(desired_buy, affordable)
        buy = max(0, buy)
    elif land_price >= sell_if_ge:
        keep_acres = int(sell_keep_app * pop)
        # Don't sell below what people can farm if we can help it
        keep_acres = max(keep_acres, min(acres, pop * HamurabiGame.ACRES_PER_PERSON))
        sell = max(0, acres - keep_acres)
        # Avoid selling so much that we can't sustain; cap at half in extreme cases
        sell = min(sell, acres)

    # Ensure trade is feasible
    if buy > 0:
        buy = min(buy, bushels // max(1, land_price))
        sell = 0
    elif sell > 0:
        sell = min(sell, acres)
        buy = 0

    bushels_after_trade = bushels - buy * land_price + sell * land_price
    acres_after_trade = acres + buy - sell

    # Feeding
    desired_feed = int(feed_mult * pop * HamurabiGame.BUSHELS_PER_PERSON)
    # Always try to be able to feed everyone if possible as a safety floor
    min_safe_feed = min(bushels_after_trade, pop * HamurabiGame.BUSHELS_PER_PERSON)
    feed = max(min_safe_feed, min(bushels_after_trade, desired_feed))
    feed = max(0, min(feed, bushels_after_trade))

    # Planting
    remaining = bushels_after_trade - feed
    reserve = int(reserve_pp * pop)
    plant_budget_bushels = max(0, remaining - reserve)
    max_by_seed = plant_budget_bushels * 2
    max_by_land = acres_after_trade
    max_by_labor = pop * HamurabiGame.ACRES_PER_PERSON
    feasible = max(0, min(max_by_seed, max_by_land, max_by_labor))
    plant = int(feasible * plant_frac)
    plant = max(0, min(plant, feasible))

    # If we reserved too much and can still plant more safely, allow some planting
    if plant == 0 and remaining > 0:
        feasible2 = max(0, min(remaining * 2, max_by_land, max_by_labor))
        plant = int(feasible2 * min(plant_frac, 0.5))

    action = (buy, sell, feed, plant)
    ok, _ = game.validate_turn(*action, land_price)
    if ok:
        return action

    # Fallback: conservative valid action
    return conservative_action(game, land_price)


def conservative_action(game: HamurabiGame, land_price: int) -> Action:
    s = game.state
    feed = min(s.bushels, s.population * HamurabiGame.BUSHELS_PER_PERSON)
    remaining = s.bushels - feed
    plant = min(s.acres, s.population * HamurabiGame.ACRES_PER_PERSON, remaining * 2)
    action = (0, 0, feed, max(0, plant))
    ok, _ = game.validate_turn(*action, land_price)
    if ok:
        return action
    return (0, 0, min(s.bushels, s.population * HamurabiGame.BUSHELS_PER_PERSON), 0)


@dataclass
class PlayResult:
    completed: bool
    survived: bool
    impeached: bool
    years_played: int
    final_population: int
    final_acres: int
    total_starved: int
    avg_starvation_pct: float
    score: float


def play_game(genome: Genome, seed: int) -> PlayResult:
    rng = random.Random(seed)
    game = HamurabiGame(rng=rng)
    g = clamp_genome(genome)

    while not game.state.game_over:
        land_price = game.current_land_price()
        action = decode_action(g, game, land_price)
        ok, _ = game.validate_turn(*action, land_price)
        if not ok:
            action = conservative_action(game, land_price)
        game.advance_year(*action, land_price)

    st = game.state
    completed = st.game_over and not st.impeached and st.population > 0 and st.years_played >= HamurabiGame.MAX_YEARS
    survived = not st.impeached and st.population > 0
    app = (st.acres / st.population) if st.population > 0 else 0.0
    avg_starv = st.cumulative_starvation_pct / max(1, st.years_played)

    # Fitness/score emphasizing completion and survival
    score = 0.0
    score += 120.0 if completed else 0.0
    score += 20.0 if survived else 0.0
    score -= 120.0 if st.impeached else 0.0
    score += st.years_played * 2.0
    score += st.population * 0.15
    score += app * 2.0
    score -= st.total_starved * 0.4
    score -= avg_starv * 0.8
    if st.population <= 0:
        score -= 30.0

    return PlayResult(
        completed=completed,
        survived=survived,
        impeached=st.impeached,
        years_played=st.years_played,
        final_population=st.population,
        final_acres=st.acres,
        total_starved=st.total_starved,
        avg_starvation_pct=avg_starv,
        score=score,
    )


class HamurabiGASearchProblem(SearchProblem):
    """SimpleAI SearchProblem for evolving Hamurabi policy genomes."""

    def __init__(self, fitness_games: int = 10, seed: int = 0):
        self.fitness_games = fitness_games
        self.rng = random.Random(seed)
        # Use a fixed base for evaluation seeds to make comparisons fairer within run;
        # still stochastic across games.
        self.eval_seed_base = self.rng.randint(0, 10**9)
        self._eval_counter = 0
        super().__init__(initial_state=None)

    def generate_random_state(self) -> Genome:
        return random_genome(self.rng)

    def crossover(self, state1: Genome, state2: Genome) -> Genome:
        # Uniform crossover with small blend
        child = []
        for i, (a, b) in enumerate(zip(state1, state2)):
            if self.rng.random() < 0.5:
                v = a
            else:
                v = b
            # occasional blend
            if self.rng.random() < 0.3:
                t = self.rng.random()
                v = t * a + (1 - t) * b
            child.append(v)
        return clamp_genome(tuple(child))

    def mutate(self, state: Genome) -> Genome:
        genes = list(state)
        # mutate 1-3 genes
        num = 1 if self.rng.random() < 0.7 else (2 if self.rng.random() < 0.8 else 3)
        idxs = self.rng.sample(range(len(genes)), k=min(num, len(genes)))
        for i in idxs:
            lo, hi = GENE_BOUNDS[i]
            span = hi - lo
            if self.rng.random() < 0.5:
                # gaussian-like perturbation
                genes[i] += self.rng.gauss(0, span * 0.15)
            else:
                # random reset
                genes[i] = self.rng.uniform(lo, hi)
        return clamp_genome(tuple(genes))

    def value(self, state: Genome) -> float:
        # Average fitness over several simulated games
        total = 0.0
        base = self.eval_seed_base + self._eval_counter * 10007
        self._eval_counter += 1
        for k in range(self.fitness_games):
            res = play_game(state, seed=base + k)
            total += res.score
        return total / self.fitness_games


def evaluate_genome(genome: Genome, games: int = 200, seed: int = 12345) -> dict:
    rng = random.Random(seed)
    results = [play_game(genome, rng.randint(0, 10**9)) for _ in range(games)]
    n = len(results)
    completed = sum(1 for r in results if r.completed)
    survived = sum(1 for r in results if r.survived)
    impeached = sum(1 for r in results if r.impeached)
    avg_score = sum(r.score for r in results) / n
    avg_pop = sum(r.final_population for r in results) / n
    avg_acres = sum(r.final_acres for r in results) / n
    avg_starved = sum(r.total_starved for r in results) / n
    avg_years = sum(r.years_played for r in results) / n
    avg_starv_pct = sum(r.avg_starvation_pct for r in results) / n
    avg_app = sum((r.final_acres / r.final_population) if r.final_population > 0 else 0.0 for r in results) / n
    return {
        "games": n,
        "completion_rate": completed / n,
        "survival_rate": survived / n,
        "impeachment_rate": impeached / n,
        "avg_score": avg_score,
        "avg_final_population": avg_pop,
        "avg_final_acres": avg_acres,
        "avg_total_starved": avg_starved,
        "avg_years_played": avg_years,
        "avg_starvation_pct": avg_starv_pct,
        "avg_acres_per_person": avg_app,
    }


def format_genome(genome: Genome) -> str:
    names = [
        "feed_mult", "reserve_per_person", "plant_fraction",
        "buy_if_price_le", "buy_target_app", "sell_if_price_ge", "sell_keep_app",
    ]
    return ", ".join(f"{n}={v:.3f}" for n, v in zip(names, genome))


def demo_game(genome: Genome, seed: int = 7) -> PlayResult:
    rng = random.Random(seed)
    game = HamurabiGame(rng=rng)
    g = clamp_genome(genome)
    print("\n--- GA-evolved policy demo game ---")
    print(f"Genome: {format_genome(g)}")

    while not game.state.game_over:
        land_price = game.current_land_price()
        s = game.state
        before = GameState(**vars(s))
        action = decode_action(g, game, land_price)
        buy, sell, feed, plant = action
        ok, _ = game.validate_turn(*action, land_price)
        if not ok:
            action = conservative_action(game, land_price)
            buy, sell, feed, plant = action
        game.advance_year(*action, land_price)
        ns = game.state
        trade = f"bought {buy}" if buy else (f"sold {sell}" if sell else "no trade")
        print(
            f"Year {before.year}: price={land_price}, pop={before.population}, bushels={before.bushels}, acres={before.acres} -> "
            f"{trade}, fed={feed}, planted={plant} | "
            f"starved={ns.starved}, pop_now={ns.population}, bushels_now={ns.bushels}, acres_now={ns.acres}"
        )
        if ns.plague:
            print("  (plague struck)")

    st = game.state
    completed = st.game_over and not st.impeached and st.population > 0 and st.years_played >= HamurabiGame.MAX_YEARS
    print(f"\nOutcome: {'Completed term' if completed else ('Impeached' if st.impeached else 'Failed')}")
    print(f"Years played: {st.years_played}, final pop: {st.population}, final acres: {st.acres}, total starved: {st.total_starved}")
    if st.end_message:
        print("Assessment:")
        print(st.end_message)
    return play_game(genome, seed)


def evolve(
    population_size: int = 40,
    mutation_chance: float = 0.2,
    iterations_limit: int = 30,
    fitness_games: int = 10,
    seed: int = 42,
    verbose: bool = True,
):
    if verbose:
        print("Hamurabi Genetic Algorithm (SimpleAI genetic search)")
        print(f"  population_size={population_size}, mutation_chance={mutation_chance}, iterations={iterations_limit}, fitness_games={fitness_games}, seed={seed}")
    problem = HamurabiGASearchProblem(fitness_games=fitness_games, seed=seed)
    result = genetic(
        problem,
        population_size=population_size,
        mutation_chance=mutation_chance,
        iterations_limit=iterations_limit,
    )
    best_genome = clamp_genome(result.state)
    best_fitness = result.value
    if verbose:
        print(f"\nBest genome: {format_genome(best_genome)}")
        print(f"Best fitness (avg over {fitness_games} games during search): {best_fitness:.2f}")
    return best_genome, best_fitness


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolve a Hamurabi-playing policy using a genetic algorithm (SimpleAI).")
    parser.add_argument("--population", type=int, default=40, help="GA population size (default: 40)")
    parser.add_argument("--mutation", type=float, default=0.2, help="Mutation chance (default: 0.2)")
    parser.add_argument("--iterations", type=int, default=30, help="GA iterations/generations limit (default: 30)")
    parser.add_argument("--fitness-games", type=int, default=10, help="Games per fitness evaluation (default: 10)")
    parser.add_argument("--eval-games", type=int, default=200, help="Games for final evaluation (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--demo-seed", type=int, default=7, help="Seed for demo game (default: 7)")
    args = parser.parse_args()

    best_genome, best_fitness = evolve(
        population_size=args.population,
        mutation_chance=args.mutation,
        iterations_limit=args.iterations,
        fitness_games=args.fitness_games,
        seed=args.seed,
        verbose=True,
    )

    print("\nFinal evaluation (greedy policy, no training noise):")
    metrics = evaluate_genome(best_genome, games=args.eval_games, seed=12345)
    print(f"  games={metrics['games']}")
    print(f"  completion_rate={metrics['completion_rate']*100:.1f}%")
    print(f"  survival_rate={metrics['survival_rate']*100:.1f}%")
    print(f"  impeachment_rate={metrics['impeachment_rate']*100:.1f}%")
    print(f"  avg_years_played={metrics['avg_years_played']:.2f}")
    print(f"  avg_final_population={metrics['avg_final_population']:.1f}")
    print(f"  avg_final_acres={metrics['avg_final_acres']:.1f}")
    print(f"  avg_acres_per_person={metrics['avg_acres_per_person']:.2f}")
    print(f"  avg_total_starved={metrics['avg_total_starved']:.1f}")
    print(f"  avg_starvation_pct={metrics['avg_starvation_pct']:.2f}")
    print(f"  avg_fitness_score={metrics['avg_score']:.2f}")

    demo_game(best_genome, seed=args.demo_seed)


if __name__ == "__main__":
    main()
