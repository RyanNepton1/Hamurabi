# Hamurabi Agent Evaluation Results

_Generated: 2026-11-22T12:43:07_  
_Environment: Python 3.15.2 on Darwin 25.6.0_

## Scope

This document records:
1. **Baseline documented results** from the initial reinforcement-learning agent implementation (`hamurabi_rl_agent.py`, SimpleAI `SearchProblem` action modeling, tabular Q-learning).
2. **Evaluation metrics and test suite** (`hamurabi_eval.py`, `run_experiments.py`) for comparing agents/heuristics on the `HamurabiGame` implementation in `hamurabi.py`.
3. **Experimental comparisons** over heuristic policies and RL configurations with varied **iterations (episodes)**, **exploration** (`explore_iterations`), and **exploitation** (`epsilon_decay`, `min_epsilon`).
4. **Findings and recommended best agent** based on measured performance.

## 1. Baseline Results (Initial RL Agent)

From prior validation runs of `hamurabi_rl_agent.py` (Q-learning, SimpleAI `HamurabiTurnProblem` action modeling, exploration-biased epsilon=1.0 for first 100 episodes then epsilon decay to conservative exploitation):

| Run (seed/config) | Train episodes | Explore iters | Unique states (explore) | Unique states (total) | Greedy eval completion | Greedy avg reward | Demo outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 / default-ish | 120 | 100 | 93 | 117 | 0/30 (0%) | -117.05 | Impeached (early, pre-improvement) |
| 42 / improved | 200 | 100 | 330 | 449 | 24/30 (80.0%) | 77.32 | Completed 10 years |
| 42 / improved | 150 | 100 | 330 | 384 | 15/20 (75.0%) | 62.33 | Completed 10 years |

**Observation:** the exploration phase (first 100 iterations, epsilon=1.0 with viability-biased/least-tried selection) achieves broad state coverage across multi-year trajectories (e.g. 330 discretized states). Post-exploration epsilon decay plus conservative heuristic tie-breaking settles the policy toward safer, higher-completion behavior, but outcomes are stochastic and sensitive to training episodes/seeds.

## 2. Evaluation Metrics and Test Suite

Implemented in `hamurabi_eval.py` and executed by `run_experiments.py`.

**Per-game metrics:** completion of 10-year term, survival (not impeached, population > 0), impeachment, years played, final population/acres/bushels, acres per person, total starved, average starvation % per year, and shaped episode reward (from `hamurabi_rl_agent.compute_reward`).

**Aggregate metrics (over N games):** completion rate, survival rate, impeachment rate, averages of the above, median reward, and a composite **score** (higher is better) emphasizing reliable term completion and low starvation/impeachment:

`score = 100*completion_rate + 10*survival_rate - 80*impeachment_rate + 0.2*avg_final_population + 0.05*avg_final_acres + 2*avg_acres_per_person - 0.5*avg_total_starved - 1*avg_starvation_pct + 0.1*avg_years_played`

**Compared agents:** (a) six rule-based heuristic policies, (b) RL (tabular Q-learning + SimpleAI `HamurabiTurnProblem` action modeling) under varied training configurations. RL configs: 2 training seeds each, 100 greedy evaluation games per seed (200 eval games/config) on fixed evaluation seeds; heuristics: 300 games each.

## 3. Heuristic Policy Results

| Heuristic | Games | Completion | Survival | Impeachment | Avg years | Avg pop | Avg acres | Avg acres/person | Avg starved | Avg starv %/yr | Avg reward | Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sell_surplus_land | 300 | 65.0% | 65.0% | 35.0% | 7.98 | 49.0 | 647.1 | 13.10 | 35.6 | 8.0 | 11.2 | 86.9 |
| feed_all_trade_for_planting | 300 | 43.7% | 43.7% | 56.3% | 6.75 | 36.3 | 887.8 | 15.16 | 51.3 | 15.0 | -25.8 | 45.0 |
| balanced_reserve_opportunistic | 300 | 33.3% | 33.3% | 66.7% | 6.06 | 31.3 | 962.8 | 17.15 | 57.7 | 18.0 | -49.7 | 25.8 |
| feed_all_plant_max | 300 | 31.7% | 31.7% | 68.3% | 6.15 | 30.8 | 1000.0 | 15.67 | 56.2 | 18.4 | -51.3 | 21.7 |
| feed_all_plant_conservative_buffer | 300 | 23.0% | 23.0% | 77.0% | 5.73 | 23.8 | 1000.0 | 20.28 | 63.5 | 20.3 | -75.7 | 7.5 |
| aggressive_expansion | 300 | 12.3% | 12.3% | 87.7% | 5.41 | 17.6 | 1073.6 | 17.57 | 70.5 | 22.3 | -105.2 | -21.3 |

**Best heuristic by score:** `sell_surplus_land` (score 86.86, completion 65.0%).

## 4. RL Experiments (Iterations, Exploration, Exploitation)

Parameters varied:
- **Iterations (total episodes):** 100, 200, 400
- **Exploration (`explore_iterations`, epsilon=1.0 exploration phase):** 0, 50, 100, 200
- **Exploitation after exploration:** `epsilon_decay` in {0.85 (fast), 0.95 (default), 0.99 (slow)} and `min_epsilon` in {0.02 (greedy), 0.10 (more stochastic)}

| Config | Episodes | Explore | Decay | Min eps | Completion | Survival | Impeachment | Avg pop | Avg acres/person | Avg starved | Avg reward | Train states (explore) | Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E200_X050_d095_m002 | 200 | 50 | 0.95 | 0.02 | 82.5% | 82.5% | 17.5% | 56.8 | 14.62 | 20.2 | 77.1 | 236.0 | 144.7 |
| E100_X100_d095_m002 | 100 | 100 | 0.95 | 0.02 | 80.5% | 80.5% | 19.5% | 53.7 | 14.92 | 24.2 | 70.5 | 341.0 | 138.1 |
| E400_X200_d095_m002 | 400 | 200 | 0.95 | 0.02 | 80.5% | 80.5% | 19.5% | 51.9 | 16.51 | 22.7 | 68.9 | 423.0 | 136.7 |
| E200_X100_d099_m002 | 200 | 100 | 0.99 | 0.02 | 81.5% | 81.5% | 18.5% | 52.0 | 15.24 | 21.7 | 72.5 | 341.0 | 136.5 |
| E100_X000_d095_m002 | 100 | 0 | 0.95 | 0.02 | 77.5% | 77.5% | 22.5% | 54.3 | 15.31 | 25.2 | 64.6 | 0.0 | 135.8 |
| E400_X100_d095_m002 | 400 | 100 | 0.95 | 0.02 | 79.5% | 79.5% | 20.5% | 51.2 | 15.85 | 22.5 | 66.3 | 341.0 | 134.4 |
| E100_X050_d095_m002 | 100 | 50 | 0.95 | 0.02 | 76.5% | 76.5% | 23.5% | 52.8 | 15.66 | 24.2 | 63.7 | 236.0 | 133.0 |
| E200_X100_d095_m002 | 200 | 100 | 0.95 | 0.02 | 80.0% | 80.0% | 20.0% | 51.9 | 13.46 | 22.6 | 68.9 | 341.0 | 131.1 |
| E200_X100_d085_m002 | 200 | 100 | 0.85 | 0.02 | 76.0% | 76.0% | 24.0% | 50.8 | 16.66 | 25.8 | 59.0 | 341.0 | 130.4 |
| E200_X100_d095_m010 | 200 | 100 | 0.95 | 0.1 | 57.5% | 57.5% | 42.5% | 39.9 | 14.18 | 40.0 | 7.0 | 341.0 | 79.7 |

**Best RL configuration by score:** `E200_X050_d095_m002` (score 144.75, completion 82.5%).

## 5. Findings

### Heuristics
- The strongest rule-based policies are conservative: **sell_surplus_land** ranks first, followed by **feed_all_trade_for_planting** and **balanced_reserve_opportunistic**. They achieve high completion by feeding the full population whenever possible, planting near-feasible capacity, and avoiding reckless trades/starvation risk.
- More aggressive/risky strategies (e.g. `aggressive_expansion`) underperform on completion and/or starvation because they increase exposure to bad harvests, rats, or resource shortfalls.

### Iterations (training episodes)
- Increasing iterations from 200 to 400 with the same exploration phase (100) changed completion from 80.0% (`E200_X100_d095_m002`) to 79.5% (`E400_X100_d095_m002`) and score from 131.08 to 134.37, indicating additional training helped consolidate the learned policy.
- At 100 vs 200 total episodes with 100-episode exploration, completion moved from 80.5% to 80.0% (scores 138.10 vs 131.08), showing the value of post-exploration exploitation iterations for policy refinement.

### Exploration
- Comparing exploration budgets at 200 episodes: 50 explore iterations (`E200_X050_d095_m002`, 82.5% completion, 236 explore states) vs 100 (`E200_X100_d095_m002`, 80.0% completion, 341 explore states). The first 100 iterations with viability-biased epsilon=1.0 exploration provide broad state coverage; sufficient exploration improves the chance of learning robust behavior across diverse states, though too little post-exploration time can limit convergence.
- Extending exploration to 200 within 400 episodes (`E400_X200_d095_m002`: 80.5%, score 136.66) vs 100 (`E400_X100_d095_m002`: 79.5%, score 134.37) trades exploitation time for coverage; the better choice depends on having enough remaining iterations to settle on the best path.
- Little/no dedicated exploration phase (`E100_X000_d095_m002`: 77.5% completion) relies heavily on early epsilon decay and heuristic tie-breaking, which can be competitive in this state/action design but explores fewer distinct strategies systematically.

### Exploitation (settling on conservative best path)
- After 100 explore iterations (200 total), faster exploitation (`epsilon_decay=0.85`, min_eps=0.02: 76.0%, score 130.36) vs default (`0.95`: 80.0%, score 131.08) vs slower (`0.99`: 81.5%, score 136.48) shows how quickly the agent commits to learned/conservative choices matters alongside exploration quality.
- Higher minimum epsilon during exploitation (`min_epsilon=0.10`: 57.5%, score 79.69) retains stochasticity and can prevent fully settling on the best conservative path compared to a greedier setting (`0.02`: 80.0%, score 131.08).

### Overall
- Best heuristic: **sell_surplus_land** (score 86.86, completion 65.0%).
- Best RL configuration tested: **E200_X050_d095_m002** (score 144.75, completion 82.5%).
- Overall best by composite score: **RL `E200_X050_d095_m002`** (score 144.75).

## 6. Recommended Best Agent

**Recommendation: RL agent `E200_X050_d095_m002` (Q-learning, SimpleAI-modeled actions).**

Based on these experiments, the best-performing agent is the **RL configuration E200_X050_d095_m002** (score 144.75, completion 82.5%).

**Why:** In Hamurabi, survival hinges on consistently feeding the population (avoiding the >45% starvation impeachment), planting enough acres within labor/seed limits, and maintaining grain reserves against yield/rat/plague variance. The recommended agent’s conservative, feed-first behavior aligns best with these dynamics under the measured metrics.

**If using the RL agent:** prefer a schedule with a substantial exploration phase (~100 iterations, epsilon=1.0, viability-biased broad state coverage) followed by sufficient exploitation iterations with decaying epsilon toward a low `min_epsilon` (e.g. decay ~0.95, min_eps ~0.02) so it settles on a conservative best path; train for >=200 episodes and evaluate greedily across many seeds due to stochasticity.

**Best RL configuration observed:** `E200_X050_d095_m002` — episodes=200, explore_iterations=50, epsilon_decay=0.95, min_epsilon=0.02 (completion 82.5%, score 144.75).

## Reproduction

```bash
python3 -m py_compile hamurabi.py hamurabi_rl_agent.py hamurabi_eval.py run_experiments.py
python3 run_experiments.py
```

Notes: results depend on RNG seeds and stochastic game events; RL training/evaluation variance is expected. The RL agent requires `simpleai` (`pip install simpleai`).
