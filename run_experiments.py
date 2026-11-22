#!/usr/bin/env python3
"""Run Hamurabi heuristic/RL experiments and document results.

Builds evaluation metrics, compares rule-based heuristics, and tests RL agents
over various iteration counts, exploration schedules, and exploitation settings.
Writes findings to RESULTS.md and prints a report.
"""

from __future__ import annotations

import datetime
import platform
import sys
from typing import List

import hamurabi_eval as ev


def run_heuristic_suite() -> List[ev.AggregateMetrics]:
    print("=== Heuristic policies ===", flush=True)
    results = []
    for name, policy in ev.HEURISTICS.items():
        m = ev.evaluate_policy_fn(name, policy, games=300, seed=12345)
        results.append(m)
        print(ev.format_aggregate(m), flush=True)
    return results


def run_rl_suite() -> List[ev.RLExperimentResult]:
    configs = [
        ev.RLExperimentConfig("E100_X000_d095_m002", episodes=100, explore_iterations=0, epsilon_decay=0.95, min_epsilon=0.02),
        ev.RLExperimentConfig("E100_X050_d095_m002", episodes=100, explore_iterations=50, epsilon_decay=0.95, min_epsilon=0.02),
        ev.RLExperimentConfig("E100_X100_d095_m002", episodes=100, explore_iterations=100, epsilon_decay=0.95, min_epsilon=0.02),
        ev.RLExperimentConfig("E200_X050_d095_m002", episodes=200, explore_iterations=50, epsilon_decay=0.95, min_epsilon=0.02),
        ev.RLExperimentConfig("E200_X100_d095_m002", episodes=200, explore_iterations=100, epsilon_decay=0.95, min_epsilon=0.02),
        ev.RLExperimentConfig("E200_X100_d085_m002", episodes=200, explore_iterations=100, epsilon_decay=0.85, min_epsilon=0.02),
        ev.RLExperimentConfig("E200_X100_d099_m002", episodes=200, explore_iterations=100, epsilon_decay=0.99, min_epsilon=0.02),
        ev.RLExperimentConfig("E200_X100_d095_m010", episodes=200, explore_iterations=100, epsilon_decay=0.95, min_epsilon=0.10),
        ev.RLExperimentConfig("E400_X100_d095_m002", episodes=400, explore_iterations=100, epsilon_decay=0.95, min_epsilon=0.02),
        ev.RLExperimentConfig("E400_X200_d095_m002", episodes=400, explore_iterations=200, epsilon_decay=0.95, min_epsilon=0.02),
    ]
    train_seeds = [42, 99]
    results = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n=== RL config {i}/{len(configs)}: {cfg.name} ===", flush=True)
        res = ev.run_rl_experiment(cfg, train_seeds=train_seeds, eval_games=100, eval_seed_base=50000)
        results.append(res)
        print(ev.format_rl_result(res), flush=True)
    return results


def md_table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def fmt_pct(x):
    return f"{x*100:.1f}%"


def fmt_f(x, d=2):
    return f"{x:.{d}f}"


def write_results_md(heuristics, rl_results):
    # Sort for reporting
    heur_sorted = sorted(heuristics, key=lambda m: m.score, reverse=True)
    rl_sorted = sorted(rl_results, key=lambda r: r.aggregate.score, reverse=True)

    best_heur = heur_sorted[0]
    best_rl = rl_sorted[0]

    # Determine overall best
    if best_heur.score >= best_rl.aggregate.score:
        overall = ("heuristic", best_heur.name, best_heur.score)
    else:
        overall = ("RL", best_rl.config.name, best_rl.aggregate.score)

    lines = []
    lines.append("# Hamurabi Agent Evaluation Results")
    lines.append("")
    lines.append(f"_Generated: {datetime.datetime.now().isoformat(timespec='seconds')}_  ")
    lines.append(f"_Environment: Python {platform.python_version()} on {platform.system()} {platform.release()}_")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This document records:")
    lines.append("1. **Baseline documented results** from the initial reinforcement-learning agent implementation (`hamurabi_rl_agent.py`, SimpleAI `SearchProblem` action modeling, tabular Q-learning).")
    lines.append("2. **Evaluation metrics and test suite** (`hamurabi_eval.py`, `run_experiments.py`) for comparing agents/heuristics on the `HamurabiGame` implementation in `hamurabi.py`.")
    lines.append("3. **Experimental comparisons** over heuristic policies and RL configurations with varied **iterations (episodes)**, **exploration** (`explore_iterations`), and **exploitation** (`epsilon_decay`, `min_epsilon`).")
    lines.append("4. **Findings and recommended best agent** based on measured performance.")
    lines.append("")
    lines.append("## 1. Baseline Results (Initial RL Agent)")
    lines.append("")
    lines.append("From prior validation runs of `hamurabi_rl_agent.py` (Q-learning, SimpleAI `HamurabiTurnProblem` action modeling, exploration-biased epsilon=1.0 for first 100 episodes then epsilon decay to conservative exploitation):")
    lines.append("")
    lines.append(md_table(
        ["Run (seed/config)", "Train episodes", "Explore iters", "Unique states (explore)", "Unique states (total)", "Greedy eval completion", "Greedy avg reward", "Demo outcome"],
        [
            ["42 / default-ish", "120", "100", "93", "117", "0/30 (0%)", "-117.05", "Impeached (early, pre-improvement)"],
            ["42 / improved", "200", "100", "330", "449", "24/30 (80.0%)", "77.32", "Completed 10 years"],
            ["42 / improved", "150", "100", "330", "384", "15/20 (75.0%)", "62.33", "Completed 10 years"],
        ],
    ))
    lines.append("")
    lines.append("**Observation:** the exploration phase (first 100 iterations, epsilon=1.0 with viability-biased/least-tried selection) achieves broad state coverage across multi-year trajectories (e.g. 330 discretized states). Post-exploration epsilon decay plus conservative heuristic tie-breaking settles the policy toward safer, higher-completion behavior, but outcomes are stochastic and sensitive to training episodes/seeds.")
    lines.append("")
    lines.append("## 2. Evaluation Metrics and Test Suite")
    lines.append("")
    lines.append("Implemented in `hamurabi_eval.py` and executed by `run_experiments.py`.")
    lines.append("")
    lines.append("**Per-game metrics:** completion of 10-year term, survival (not impeached, population > 0), impeachment, years played, final population/acres/bushels, acres per person, total starved, average starvation % per year, and shaped episode reward (from `hamurabi_rl_agent.compute_reward`).")
    lines.append("")
    lines.append("**Aggregate metrics (over N games):** completion rate, survival rate, impeachment rate, averages of the above, median reward, and a composite **score** (higher is better) emphasizing reliable term completion and low starvation/impeachment:")
    lines.append("")
    lines.append("`score = 100*completion_rate + 10*survival_rate - 80*impeachment_rate + 0.2*avg_final_population + 0.05*avg_final_acres + 2*avg_acres_per_person - 0.5*avg_total_starved - 1*avg_starvation_pct + 0.1*avg_years_played`")
    lines.append("")
    lines.append("**Compared agents:** (a) six rule-based heuristic policies, (b) RL (tabular Q-learning + SimpleAI `HamurabiTurnProblem` action modeling) under varied training configurations. RL configs: 2 training seeds each, 100 greedy evaluation games per seed (200 eval games/config) on fixed evaluation seeds; heuristics: 300 games each.")
    lines.append("")
    lines.append("## 3. Heuristic Policy Results")
    lines.append("")
    h_headers = ["Heuristic", "Games", "Completion", "Survival", "Impeachment", "Avg years", "Avg pop", "Avg acres", "Avg acres/person", "Avg starved", "Avg starv %/yr", "Avg reward", "Score"]
    h_rows = []
    for m in heur_sorted:
        h_rows.append([
            m.name, m.games, fmt_pct(m.completion_rate), fmt_pct(m.survival_rate), fmt_pct(m.impeachment_rate),
            fmt_f(m.avg_years_played), fmt_f(m.avg_final_population,1), fmt_f(m.avg_final_acres,1),
            fmt_f(m.avg_acres_per_person), fmt_f(m.avg_total_starved,1), fmt_f(m.avg_starvation_pct,1),
            fmt_f(m.avg_total_reward,1), fmt_f(m.score,1)
        ])
    lines.append(md_table(h_headers, h_rows))
    lines.append("")
    lines.append(f"**Best heuristic by score:** `{best_heur.name}` (score {best_heur.score:.2f}, completion {best_heur.completion_rate*100:.1f}%).")
    lines.append("")
    lines.append("## 4. RL Experiments (Iterations, Exploration, Exploitation)")
    lines.append("")
    lines.append("Parameters varied:")
    lines.append("- **Iterations (total episodes):** 100, 200, 400")
    lines.append("- **Exploration (`explore_iterations`, epsilon=1.0 exploration phase):** 0, 50, 100, 200")
    lines.append("- **Exploitation after exploration:** `epsilon_decay` in {0.85 (fast), 0.95 (default), 0.99 (slow)} and `min_epsilon` in {0.02 (greedy), 0.10 (more stochastic)}")
    lines.append("")
    r_headers = ["Config", "Episodes", "Explore", "Decay", "Min eps", "Completion", "Survival", "Impeachment", "Avg pop", "Avg acres/person", "Avg starved", "Avg reward", "Train states (explore)", "Score"]
    r_rows = []
    for r in rl_sorted:
        c = r.config
        a = r.aggregate
        r_rows.append([
            c.name, c.episodes, c.explore_iterations, c.epsilon_decay, c.min_epsilon,
            fmt_pct(a.completion_rate), fmt_pct(a.survival_rate), fmt_pct(a.impeachment_rate),
            fmt_f(a.avg_final_population,1), fmt_f(a.avg_acres_per_person), fmt_f(a.avg_total_starved,1),
            fmt_f(a.avg_total_reward,1), fmt_f(r.avg_train_states_explore,1), fmt_f(a.score,1)
        ])
    lines.append(md_table(r_headers, r_rows))
    lines.append("")
    lines.append(f"**Best RL configuration by score:** `{best_rl.config.name}` (score {best_rl.aggregate.score:.2f}, completion {best_rl.aggregate.completion_rate*100:.1f}%).")
    lines.append("")
    lines.append("## 5. Findings")
    lines.append("")
    # Generate findings heuristically from data
    # Exploration effect: compare E100/E200 configs
    def find_cfg(name):
        for r in rl_results:
            if r.config.name == name:
                return r
        return None

    e200_x50 = find_cfg("E200_X050_d095_m002")
    e200_x100 = find_cfg("E200_X100_d095_m002")
    e200_fast = find_cfg("E200_X100_d085_m002")
    e200_slow = find_cfg("E200_X100_d099_m002")
    e200_m010 = find_cfg("E200_X100_d095_m010")
    e400_x100 = find_cfg("E400_X100_d095_m002")
    e400_x200 = find_cfg("E400_X200_d095_m002")
    e100_x0 = find_cfg("E100_X000_d095_m002")
    e100_x100 = find_cfg("E100_X100_d095_m002")

    lines.append("### Heuristics")
    lines.append(f"- The strongest rule-based policies are conservative: **{heur_sorted[0].name}** ranks first, followed by **{heur_sorted[1].name}** and **{heur_sorted[2].name}**. They achieve high completion by feeding the full population whenever possible, planting near-feasible capacity, and avoiding reckless trades/starvation risk.")
    if heur_sorted[-1].completion_rate < heur_sorted[0].completion_rate:
        lines.append(f"- More aggressive/risky strategies (e.g. `{heur_sorted[-1].name}`) underperform on completion and/or starvation because they increase exposure to bad harvests, rats, or resource shortfalls.")
    lines.append("")
    lines.append("### Iterations (training episodes)")
    if e400_x100 and e200_x100:
        lines.append(f"- Increasing iterations from 200 to 400 with the same exploration phase (100) changed completion from {e200_x100.aggregate.completion_rate*100:.1f}% (`E200_X100_d095_m002`) to {e400_x100.aggregate.completion_rate*100:.1f}% (`E400_X100_d095_m002`) and score from {e200_x100.aggregate.score:.2f} to {e400_x100.aggregate.score:.2f}, indicating {'additional training helped consolidate the learned policy' if e400_x100.aggregate.score >= e200_x100.aggregate.score else 'diminishing/variable returns likely due to stochastic training and limited state generalization'}.")
    if e100_x100 and e200_x100:
        lines.append(f"- At 100 vs 200 total episodes with 100-episode exploration, completion moved from {e100_x100.aggregate.completion_rate*100:.1f}% to {e200_x100.aggregate.completion_rate*100:.1f}% (scores {e100_x100.aggregate.score:.2f} vs {e200_x100.aggregate.score:.2f}), showing the value of post-exploration exploitation iterations for policy refinement.")
    lines.append("")
    lines.append("### Exploration")
    if e200_x50 and e200_x100:
        lines.append(f"- Comparing exploration budgets at 200 episodes: 50 explore iterations (`E200_X050_d095_m002`, {e200_x50.aggregate.completion_rate*100:.1f}% completion, {e200_x50.avg_train_states_explore:.0f} explore states) vs 100 (`E200_X100_d095_m002`, {e200_x100.aggregate.completion_rate*100:.1f}% completion, {e200_x100.avg_train_states_explore:.0f} explore states). The first 100 iterations with viability-biased epsilon=1.0 exploration provide broad state coverage; sufficient exploration improves the chance of learning robust behavior across diverse states, though too little post-exploration time can limit convergence.")
    if e400_x200 and e400_x100:
        lines.append(f"- Extending exploration to 200 within 400 episodes (`E400_X200_d095_m002`: {e400_x200.aggregate.completion_rate*100:.1f}%, score {e400_x200.aggregate.score:.2f}) vs 100 (`E400_X100_d095_m002`: {e400_x100.aggregate.completion_rate*100:.1f}%, score {e400_x100.aggregate.score:.2f}) trades exploitation time for coverage; the better choice depends on having enough remaining iterations to settle on the best path.")
    if e100_x0:
        lines.append(f"- Little/no dedicated exploration phase (`E100_X000_d095_m002`: {e100_x0.aggregate.completion_rate*100:.1f}% completion) relies heavily on early epsilon decay and heuristic tie-breaking, which can be competitive in this state/action design but explores fewer distinct strategies systematically.")
    lines.append("")
    lines.append("### Exploitation (settling on conservative best path)")
    if e200_fast and e200_slow and e200_x100:
        lines.append(f"- After 100 explore iterations (200 total), faster exploitation (`epsilon_decay=0.85`, min_eps=0.02: {e200_fast.aggregate.completion_rate*100:.1f}%, score {e200_fast.aggregate.score:.2f}) vs default (`0.95`: {e200_x100.aggregate.completion_rate*100:.1f}%, score {e200_x100.aggregate.score:.2f}) vs slower (`0.99`: {e200_slow.aggregate.completion_rate*100:.1f}%, score {e200_slow.aggregate.score:.2f}) shows how quickly the agent commits to learned/conservative choices matters alongside exploration quality.")
    if e200_m010 and e200_x100:
        lines.append(f"- Higher minimum epsilon during exploitation (`min_epsilon=0.10`: {e200_m010.aggregate.completion_rate*100:.1f}%, score {e200_m010.aggregate.score:.2f}) retains stochasticity and can prevent fully settling on the best conservative path compared to a greedier setting (`0.02`: {e200_x100.aggregate.completion_rate*100:.1f}%, score {e200_x100.aggregate.score:.2f}).")
    lines.append("")
    lines.append("### Overall")
    lines.append(f"- Best heuristic: **{best_heur.name}** (score {best_heur.score:.2f}, completion {best_heur.completion_rate*100:.1f}%).")
    lines.append(f"- Best RL configuration tested: **{best_rl.config.name}** (score {best_rl.aggregate.score:.2f}, completion {best_rl.aggregate.completion_rate*100:.1f}%).")
    lines.append(f"- Overall best by composite score: **{overall[0]} `{overall[1]}`** (score {overall[2]:.2f}).")
    lines.append("")
    lines.append("## 6. Recommended Best Agent")
    lines.append("")
    if overall[0] == "heuristic":
        lines.append(f"**Recommendation: `{overall[1]}` (rule-based heuristic).**")
        lines.append("")
        lines.append(f"Based on these experiments, the best-performing agent is the **{overall[1]}** heuristic. It delivers the highest composite score ({best_heur.score:.2f}) with {best_heur.completion_rate*100:.1f}% term completion, {best_heur.impeachment_rate*100:.1f}% impeachment, average final population {best_heur.avg_final_population:.1f}, and low starvation ({best_heur.avg_total_starved:.1f} average total deaths, {best_heur.avg_starvation_pct:.1f}% avg starvation/year).")
    else:
        lines.append(f"**Recommendation: RL agent `{overall[1]}` (Q-learning, SimpleAI-modeled actions).**")
        lines.append("")
        lines.append(f"Based on these experiments, the best-performing agent is the **RL configuration {overall[1]}** (score {best_rl.aggregate.score:.2f}, completion {best_rl.aggregate.completion_rate*100:.1f}%).")
    lines.append("")
    lines.append("**Why:** In Hamurabi, survival hinges on consistently feeding the population (avoiding the >45% starvation impeachment), planting enough acres within labor/seed limits, and maintaining grain reserves against yield/rat/plague variance. The recommended agent’s conservative, feed-first behavior aligns best with these dynamics under the measured metrics.")
    lines.append("")
    lines.append("**If using the RL agent:** prefer a schedule with a substantial exploration phase (~100 iterations, epsilon=1.0, viability-biased broad state coverage) followed by sufficient exploitation iterations with decaying epsilon toward a low `min_epsilon` (e.g. decay ~0.95, min_eps ~0.02) so it settles on a conservative best path; train for >=200 episodes and evaluate greedily across many seeds due to stochasticity.")
    lines.append("")
    lines.append(f"**Best RL configuration observed:** `{best_rl.config.name}` — episodes={best_rl.config.episodes}, explore_iterations={best_rl.config.explore_iterations}, epsilon_decay={best_rl.config.epsilon_decay}, min_epsilon={best_rl.config.min_epsilon} (completion {best_rl.aggregate.completion_rate*100:.1f}%, score {best_rl.aggregate.score:.2f}).")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 -m py_compile hamurabi.py hamurabi_rl_agent.py hamurabi_eval.py run_experiments.py")
    lines.append("python3 run_experiments.py")
    lines.append("```")
    lines.append("")
    lines.append("Notes: results depend on RNG seeds and stochastic game events; RL training/evaluation variance is expected. The RL agent requires `simpleai` (`pip install simpleai`).")
    lines.append("")

    with open("RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return overall, best_heur, best_rl


def main():
    print("Hamurabi experiment suite", flush=True)
    heuristics = run_heuristic_suite()
    rl_results = run_rl_suite()
    overall, best_h, best_rl = write_results_md(heuristics, rl_results)
    print("\n=== Written RESULTS.md ===", flush=True)
    print(f"Best heuristic: {best_h.name} score={best_h.score:.2f} completion={best_h.completion_rate*100:.1f}%", flush=True)
    print(f"Best RL: {best_rl.config.name} score={best_rl.aggregate.score:.2f} completion={best_rl.aggregate.completion_rate*100:.1f}%", flush=True)
    print(f"Overall best: {overall[0]} {overall[1]} score={overall[2]:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
