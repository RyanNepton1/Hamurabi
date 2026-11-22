#!/usr/bin/env python3
"""Individually sweep hyperparameters mentioned for the Hamurabi RL agent.

Trains each variant with: episodes=200, explore_iterations=50, epsilon_decay=0.95, min_epsilon=0.02
and compares against baseline E200_X050_d095_m002.
"""
from __future__ import annotations

import copy
import time
import hamurabi_eval as ev
import hamurabi_rl_agent as rl


BASE_TRAIN_SEEDS = [42, 99]
BASE_EVAL_GAMES = 100
BASE_EVAL_SEED = 50000
BASE_CFG = dict(episodes=200, explore_iterations=50, epsilon_decay=0.95, min_epsilon=0.02)


def run_variant(name, **train_kwargs):
    cfg = ev.RLExperimentConfig(name, **BASE_CFG)
    # ev.run_rl_experiment doesn't pass extra kwargs to rl.train, so we do it manually
    per_seed_aggs = []
    all_eval_metrics = []
    states_total = []
    states_explore = []
    q_entries = []
    for i, ts in enumerate(BASE_TRAIN_SEEDS):
        agent, _train_results, stats = rl.train(
            episodes=BASE_CFG['episodes'],
            explore_iterations=BASE_CFG['explore_iterations'],
            seed=ts,
            verbose=False,
            alpha=train_kwargs.get('alpha', 0.25),
            gamma=train_kwargs.get('gamma', 0.98),
            min_epsilon=BASE_CFG['min_epsilon'],
            epsilon_decay=BASE_CFG['epsilon_decay'],
            discretization_config=train_kwargs.get('discretization_config'),
            action_gen_config=train_kwargs.get('action_gen_config'),
            conservative_score_config=train_kwargs.get('conservative_score_config'),
            exploration_config=train_kwargs.get('exploration_config'),
            reward_config=train_kwargs.get('reward_config'),
        )
        eval_metrics = ev.evaluate_trained_agent(agent, games=BASE_EVAL_GAMES, seed=BASE_EVAL_SEED + i * 1000)
        per_seed_aggs.append(ev.aggregate_metrics(f"{name}_seed{ts}", eval_metrics))
        all_eval_metrics.extend(eval_metrics)
        states_total.append(stats.get("unique_states_total", 0))
        states_explore.append(stats.get("unique_states_explore_phase", 0))
        q_entries.append(stats.get("q_table_entries", 0))
    agg = ev.aggregate_metrics(name, all_eval_metrics)
    res = ev.RLExperimentResult(
        config=cfg,
        aggregate=agg,
        per_seed=per_seed_aggs,
        avg_train_states_total=sum(states_total)/len(states_total),
        avg_train_states_explore=sum(states_explore)/len(states_explore),
        avg_q_table_entries=sum(q_entries)/len(q_entries),
    )
    return res


def fmt(res):
    a = res.aggregate
    return (f"completion={a.completion_rate*100:5.1f}% survival={a.survival_rate*100:5.1f}% "
            f"impeach={a.impeachment_rate*100:5.1f}% avg_pop={a.avg_final_population:5.1f} "
            f"avg_reward={a.avg_total_reward:6.1f} score={a.score:6.2f} "
            f"states={res.avg_train_states_total:.0f}/{res.avg_train_states_explore:.0f}")


def main():
    t0 = time.time()
    results = []

    def add(name, **kw):
        print(f"RUN {name}", flush=True)
        t = time.time()
        r = run_variant(name, **kw)
        print(f"  -> {fmt(r)} ({time.time()-t:.1f}s)", flush=True)
        results.append(r)
        return r

    baseline = add("BASELINE_E200_X050_d095_m002")

    # Reward config sweeps
    for impeach in [60.0, 90.0, 150.0, 200.0]:
        rc = copy.deepcopy(rl.DEFAULT_REWARD_CONFIG); rc.impeach_penalty = impeach; rc.wipeout_penalty = impeach
        add(f"reward.impeach_penalty={impeach}", reward_config=rc)
    for v in [0.0, 1.0, 4.0]:
        rc = copy.deepcopy(rl.DEFAULT_REWARD_CONFIG); rc.survival_bonus = v
        add(f"reward.survival_bonus={v}", reward_config=rc)
    for v in [20.0, 30.0, 50.0, 60.0]:
        rc = copy.deepcopy(rl.DEFAULT_REWARD_CONFIG); rc.starve_penalty_scale = v
        add(f"reward.starve_penalty_scale={v}", reward_config=rc)
    for v in [2.0, 6.0, 8.0]:
        rc = copy.deepcopy(rl.DEFAULT_REWARD_CONFIG); rc.no_starve_bonus = v
        add(f"reward.no_starve_bonus={v}", reward_config=rc)
    for v in [0.0, 0.5, 2.0]:
        rc = copy.deepcopy(rl.DEFAULT_REWARD_CONFIG); rc.low_starve_bonus = v
        add(f"reward.low_starve_bonus={v}", reward_config=rc)
    for v in [0.05, 0.15, 0.2]:
        rc = copy.deepcopy(rl.DEFAULT_REWARD_CONFIG); rc.low_starve_ratio_threshold = v
        add(f"reward.low_starve_ratio_threshold={v}", reward_config=rc)
    for v in [5.0, 15.0, 20.0]:
        rc = copy.deepcopy(rl.DEFAULT_REWARD_CONFIG); rc.reserves_low_threshold = v
        add(f"reward.reserves_low_threshold={v}", reward_config=rc)
    for v in [0.5, 1.0, 3.0]:
        rc = copy.deepcopy(rl.DEFAULT_REWARD_CONFIG); rc.reserves_low_penalty = v
        add(f"reward.reserves_low_penalty={v}", reward_config=rc)
    for v in [15.0, 25.0, 30.0]:
        rc = copy.deepcopy(rl.DEFAULT_REWARD_CONFIG); rc.reserves_high_threshold = v
        add(f"reward.reserves_high_threshold={v}", reward_config=rc)
    for v in [0.3, 1.5, 3.0]:
        rc = copy.deepcopy(rl.DEFAULT_REWARD_CONFIG); rc.reserves_high_bonus = v
        add(f"reward.reserves_high_bonus={v}", reward_config=rc)
    for low, high in [(7.0,12.0),(8.0,12.0),(9.0,15.0),(10.0,15.0)]:
        rc = copy.deepcopy(rl.DEFAULT_REWARD_CONFIG); rc.app_good_low = low; rc.app_good_high = high
        add(f"reward.app_good={low}-{high}", reward_config=rc)
    for v in [0.0, 1.0]:
        rc = copy.deepcopy(rl.DEFAULT_REWARD_CONFIG); rc.app_good_bonus = v
        add(f"reward.app_good_bonus={v}", reward_config=rc)
    for v in [4.0, 7.0, 8.0]:
        rc = copy.deepcopy(rl.DEFAULT_REWARD_CONFIG); rc.app_low_threshold = v
        add(f"reward.app_low_threshold={v}", reward_config=rc)
    for v in [0.0, 1.0, 1.5]:
        rc = copy.deepcopy(rl.DEFAULT_REWARD_CONFIG); rc.app_low_penalty = v
        add(f"reward.app_low_penalty={v}", reward_config=rc)
    for v in [40.0, 60.0, 100.0, 120.0]:
        rc = copy.deepcopy(rl.DEFAULT_REWARD_CONFIG); rc.completion_tier1_bonus = v
        add(f"reward.completion_tier1_bonus={v}", reward_config=rc)
    for v in [20.0, 60.0]:
        rc = copy.deepcopy(rl.DEFAULT_REWARD_CONFIG); rc.completion_tier2_bonus = v
        add(f"reward.completion_tier2_bonus={v}", reward_config=rc)
    for v in [5.0, 25.0, 40.0]:
        rc = copy.deepcopy(rl.DEFAULT_REWARD_CONFIG); rc.completion_tier3_bonus = v
        add(f"reward.completion_tier3_bonus={v}", reward_config=rc)

    # Discretization sweeps
    for bins in [(10,20,30), (10,20,30,45,60), (15,25,35,50), (5,15,25,35,45)]:
        dc = copy.deepcopy(rl.DEFAULT_DISCRETIZATION_CONFIG); dc.bpp_bins = bins
        add(f"disc.bpp_bins={bins}", discretization_config=dc)
    for bins in [(5,8,10), (5,8,10,13,16), (6,9,12), (4,7,10,13)]:
        dc = copy.deepcopy(rl.DEFAULT_DISCRETIZATION_CONFIG); dc.app_bins = bins
        add(f"disc.app_bins={bins}", discretization_config=dc)
    for bins in [(50,100,150), (30,60,90,120), (40,80,120,170,220)]:
        dc = copy.deepcopy(rl.DEFAULT_DISCRETIZATION_CONFIG); dc.pop_bins = bins
        add(f"disc.pop_bins={bins}", discretization_config=dc)
    for bins in [(20,), (21,), (18,21,23), (19,21,23)]:
        dc = copy.deepcopy(rl.DEFAULT_DISCRETIZATION_CONFIG); dc.price_bins = bins
        add(f"disc.price_bins={bins}", discretization_config=dc)

    # Action generation sweeps
    for fracs in [(0.0,0.6,0.8,1.0), (0.0,0.5,0.75,1.0,1.1), (0.0,0.8,0.9,1.0,1.05,1.1), (0.5,0.8,1.0)]:
        ac = copy.deepcopy(rl.DEFAULT_ACTION_GEN_CONFIG); ac.feed_fracs = fracs
        add(f"action.feed_fracs={fracs}", action_gen_config=ac)
    for fracs in [(0.0,0.5,1.0), (0.0,0.25,0.5,0.75,1.0), (0.25,0.5,0.75,1.0)]:
        ac = copy.deepcopy(rl.DEFAULT_ACTION_GEN_CONFIG); ac.plant_fracs = fracs
        add(f"action.plant_fracs={fracs}", action_gen_config=ac)
    for fracs in [(0.25,0.5,1.0), (0.1,0.2,0.4,0.6,1.0), (0.1,0.5,1.0)]:
        ac = copy.deepcopy(rl.DEFAULT_ACTION_GEN_CONFIG); ac.sell_fracs = fracs
        add(f"action.sell_fracs={fracs}", action_gen_config=ac)
    for fracs in [(0.25,0.5,1.0), (0.1,0.2,0.4,0.6,1.0), (0.1,0.5,1.0)]:
        ac = copy.deepcopy(rl.DEFAULT_ACTION_GEN_CONFIG); ac.buy_fracs = fracs
        add(f"action.buy_fracs={fracs}", action_gen_config=ac)

    # Exploration sweeps
    for v in [0.9, 0.95, 1.0]:
        ec = copy.deepcopy(rl.DEFAULT_EXPLORATION_CONFIG); ec.viable_feed_ratio = v
        add(f"explore.viable_feed_ratio={v}", exploration_config=ec)
    for v in [0.6, 0.7, 0.9]:
        ec = copy.deepcopy(rl.DEFAULT_EXPLORATION_CONFIG); ec.viable_fallback_ratio = v
        add(f"explore.viable_fallback_ratio={v}", exploration_config=ec)
    for v in [0.6, 0.7, 0.9, 1.0]:
        ec = copy.deepcopy(rl.DEFAULT_EXPLORATION_CONFIG); ec.viable_pool_prob = v
        add(f"explore.viable_pool_prob={v}", exploration_config=ec)
    for v in [0.0, 0.2, 0.8, 1.0]:
        ec = copy.deepcopy(rl.DEFAULT_EXPLORATION_CONFIG); ec.heuristic_jitter = v
        add(f"explore.heuristic_jitter={v}", exploration_config=ec)

    # Conservative score sweeps
    for v in [3.0, 4.0, 8.0, 10.0]:
        cc = copy.deepcopy(rl.DEFAULT_CONSERVATIVE_SCORE_CONFIG); cc.feed_weight = v
        add(f"cons.feed_weight={v}", conservative_score_config=cc)
    for v in [0.0, 1.0, 3.0, 4.0]:
        cc = copy.deepcopy(rl.DEFAULT_CONSERVATIVE_SCORE_CONFIG); cc.full_feed_bonus = v
        add(f"cons.full_feed_bonus={v}", conservative_score_config=cc)
    for v in [1.0, 3.0, 4.0]:
        cc = copy.deepcopy(rl.DEFAULT_CONSERVATIVE_SCORE_CONFIG); cc.plant_weight = v
        add(f"cons.plant_weight={v}", conservative_score_config=cc)
    for v in [0.0, 0.5, 2.0]:
        cc = copy.deepcopy(rl.DEFAULT_CONSERVATIVE_SCORE_CONFIG); cc.sell_half_penalty = v
        add(f"cons.sell_half_penalty={v}", conservative_score_config=cc)
    for v in [0.25, 0.75, 1.0]:
        cc = copy.deepcopy(rl.DEFAULT_CONSERVATIVE_SCORE_CONFIG); cc.sell_half_threshold = v
        add(f"cons.sell_half_threshold={v}", conservative_score_config=cc)
    for v in [0.0, 1.0, 1.5]:
        cc = copy.deepcopy(rl.DEFAULT_CONSERVATIVE_SCORE_CONFIG); cc.buy_expensive_penalty = v
        add(f"cons.buy_expensive_penalty={v}", conservative_score_config=cc)
    for v in [22, 23, 25, 26]:
        cc = copy.deepcopy(rl.DEFAULT_CONSERVATIVE_SCORE_CONFIG); cc.buy_expensive_price = v
        add(f"cons.buy_expensive_price={v}", conservative_score_config=cc)
    for v in [1.0, 1.5, 2.5, 3.0]:
        cc = copy.deepcopy(rl.DEFAULT_CONSERVATIVE_SCORE_CONFIG); cc.buy_expensive_reserve_mult = v
        add(f"cons.buy_expensive_reserve_mult={v}", conservative_score_config=cc)

    # Learning params mentioned
    for v in [0.10, 0.15, 0.35, 0.50]:
        add(f"learn.alpha={v}", alpha=v)
    for v in [0.90, 0.95, 0.99, 1.00]:
        add(f"learn.gamma={v}", gamma=v)

    print("\n=== SUMMARY (sorted by score) ===")
    results_sorted = sorted(results, key=lambda r: r.aggregate.score, reverse=True)
    base_score = baseline.aggregate.score
    base_comp = baseline.aggregate.completion_rate
    for r in results_sorted:
        a = r.aggregate
        dscore = a.score - base_score
        dcomp = (a.completion_rate - base_comp) * 100
        marker = " *BASE*" if r.config.name == baseline.config.name else ""
        print(f"{r.config.name:45s} | {fmt(r)} | dScore={dscore:+6.2f} dComp={dcomp:+5.1f}pp{marker}")

    print(f"\nTOTAL_TIME={time.time()-t0:.1f}s  N={len(results)}")
    print(f"BASELINE {baseline.config.name}: score={base_score:.2f} completion={base_comp*100:.1f}%")

if __name__ == '__main__':
    main()
