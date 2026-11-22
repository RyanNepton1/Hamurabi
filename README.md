# Hamurabi

A Python implementation of the classic text-based strategy game [Hamurabi](https://en.wikipedia.org/wiki/Hamurabi_(video_game)) (originally *The Sumer Game* by Doug Dyment, popularized by David H. Ahl in *BASIC Computer Games*).

## Overview

You are Hammurabi, ruler of ancient Sumeria. Govern the kingdom for a **10-year term** by managing people, land, and grain. Each year you decide:

1. **Buy or sell land** (trading price varies from 17–26 bushels per acre each year)
2. **Feed your people** (each person needs 20 bushels per year to survive)
3. **Plant crops** (1 bushel of seed plants 2 acres; each person can farm at most 10 acres)

Random events affect each year: harvest yields vary (1–5 bushels per acre), rats may eat stored grain, immigration depends on conditions, and a plague may strike.

## Game End Conditions

- **Impeachment:** If more than 45% of the population starves in a single year, the game ends immediately.
- **Depopulation:** If the entire population dies, the game ends.
- **Term complete:** After 10 years, you receive a performance assessment based on average starvation and land per person.

## Requirements

- Python 3.8+

## Running the Game

From this directory:

```bash
python3 hamurabi.py
```

## Project Structure

- `hamurabi.py` — core game logic (`HamurabiGame`, `GameState`) and interactive CLI (`play_interactive`)
- `hamurabi_rl_agent.py` — reinforcement-learning (tabular Q-learning) agent that learns to play by interacting with `HamurabiGame`, using SimpleAI's `SearchProblem` for action modeling
- `hamurabi_eval.py` — evaluation metrics, heuristic policies, and RL experiment helpers
- `run_experiments.py` — experiment runner for heuristic/RL comparisons; generates `RESULTS.md`
- `RESULTS.md` — documented evaluation results and recommended best agent
- `hamurabi_ga.py` — genetic-algorithm policy search (SimpleAI) that interacts with the game
- `README.md` — this file

## Using the Game Logic Programmatically

The core engine is separated from I/O and supports deterministic runs via an injectable RNG:

```python
import random
from hamurabi import HamurabiGame

game = HamurabiGame(rng=random.Random(42))
price = game.current_land_price()
ok, err = game.run_year(
    acres_bought=0,
    acres_sold=0,
    bushels_fed=2000,
    acres_planted=1000,
    land_price=price,
)
print(game.state)
```

## Reinforcement-Learning Agent (SimpleAI)

`hamurabi_rl_agent.py` implements a reinforcement-based agent that learns to play Hamurabi by interacting with the `HamurabiGame` API.

- **RL method:** tabular Q-learning with epsilon-greedy action selection and reward shaping for survival/conservative rule.
- **SimpleAI usage:** candidate per-turn actions are modeled via a `simpleai.search.models.SearchProblem` subclass (`HamurabiTurnProblem`).
- **Exploration schedule:** the first 100 training iterations (episodes) run with `epsilon=1.0` and exploration biased toward viable, less-tried actions to seek broad coverage of game states; after 100 iterations epsilon decays toward a low value so the agent settles on a more conservative/best learned path (with heuristic tie-breaking for safety).
- **Interaction:** each step observes the game state/land price, selects a valid action, and applies it via `HamurabiGame.run_year` / `advance_year`.

### Run training and evaluation

```bash
python3 hamurabi_rl_agent.py --episodes 200 --explore-iterations 100 --eval-episodes 30
```

This trains the agent, prints exploration vs. post-exploration phase summaries and state-coverage stats, evaluates the greedy learned policy, and plays a demo game with the conservative policy. Options: `--episodes`, `--explore-iterations`, `--eval-episodes`, `--seed`, `--demo-seed`, `--quiet`.

Requires `simpleai` to be installed (e.g. `pip install simpleai`).

## Genetic Algorithm Agent (SimpleAI)

`hamurabi_ga.py` includes a "for fun" genetic algorithm that evolves a Hamurabi-playing policy and interacts with `HamurabiGame`.

- **GA method:** `simpleai.search.local.genetic` over a `SearchProblem` with random genome generation, crossover, mutation, and fitness evaluation.
- **Policy representation:** a 7-gene genome controlling feeding, grain reserve per person, planting fraction, and land buy/sell price thresholds and target acres-per-person.
- **Interaction:** each genome is decoded into a valid yearly action (`buy/sell/feed/plant`) with safety fallbacks; fitness is the average score over multiple stochastic simulated games.

### Run evolution and demo

```bash
python3 hamurabi_ga.py
```

Useful options:

```bash
python3 hamurabi_ga.py --population 40 --mutation 0.2 --iterations 30 --fitness-games 10 --eval-games 200 --seed 42 --demo-seed 7
```

Because Hamurabi outcomes are stochastic, evolved performance varies by seed and by the evolution/evaluation budget (larger populations, more iterations, and more fitness games generally give more stable results).

## Evaluation Experiments

`hamurabi_eval.py` and `run_experiments.py` provide evaluation metrics (completion/survival/impeachment rates, years played, population/acres, starvation, and a composite score), baseline heuristic policies, and RL configuration sweeps. See [RESULTS.md](RESULTS.md) for documented experiment results and the recommended best-performing agent from those runs.
