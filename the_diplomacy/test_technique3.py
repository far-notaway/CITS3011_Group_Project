"""
Ablation experiment for New Technique 3 (Adaptive Threat Modelling).

Compares ThreatAwareAgent(use_threat=True) vs ThreatAwareAgent(use_threat=False)
under the same fallback movement logic, across Scenario 1 (vs Static) and
Scenario 2 (vs Random/Attitude/Greedy mix), matching test.py's methodology.

Run:
    python3 test_technique3.py
"""

import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from game import run_one_game
from agent_baselines import StaticAgent, RandomAgent, GreedyAgent, AttitudeAgent
from the_diplomacy.agent_technique3_demo import ThreatAwareAgent

ALL_POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']


def scoring(centres):
    scores = {k: min(v, 18) for k, v in centres.items()}
    wins = {}
    for k, v in scores.items():
        if v == 18:
            wins[k] = 'WIN'
        elif v == 0:
            wins[k] = 'DEFEAT'
        else:
            wins[k] = 'SURVIVE'
    return scores, wins


def experiment(player_agent_factory, opponent_agent_pool, repeat_nums=10, label=''):
    all_scores = defaultdict(list)
    all_wins = defaultdict(list)
    total_games = repeat_nums * len(ALL_POWERS)

    with tqdm(total=total_games, desc=label) as pbar:
        for r in range(repeat_nums):
            for i in ALL_POWERS:
                agents_dict = {}
                for p in ALL_POWERS:
                    if p == i:
                        agents_dict[p] = player_agent_factory()
                    else:
                        opponent_agent = random.choice(opponent_agent_pool)
                        agents_dict[p] = opponent_agent()
                results, _ = run_one_game(agents_dict)
                scores, wins = scoring(results)
                all_scores[i].append(scores[i])
                all_scores['ALL'].append(scores[i])
                all_wins[i].append(wins[i])
                all_wins['ALL'].append(wins[i])
                pbar.update(1)

    scores_avg = np.mean(all_scores['ALL'])
    scores_std = np.std(all_scores['ALL'])
    win_rate = round(100 * sum(w == 'WIN' for w in all_wins['ALL']) / len(all_wins['ALL']), 2)
    survive_rate = round(100 * sum(w == 'SURVIVE' for w in all_wins['ALL']) / len(all_wins['ALL']), 2)
    defeat_rate = round(100 * sum(w == 'DEFEAT' for w in all_wins['ALL']) / len(all_wins['ALL']), 2)

    return {
        'label': label,
        'avg_sc': round(scores_avg, 2),
        'std_sc': round(scores_std, 2),
        'win_rate': win_rate,
        'survive_rate': survive_rate,
        'defeat_rate': defeat_rate,
    }


if __name__ == '__main__':
    REPEATS = 10  # bump up for the report's final numbers; kept low here for a quick run

    scenarios = {
        'Scenario 1 (vs Static)': [StaticAgent],
        'Scenario 2 (vs Random/Attitude/Greedy)': [RandomAgent, AttitudeAgent, AttitudeAgent, GreedyAgent, GreedyAgent],
    }

    results = []
    for scen_name, pool in scenarios.items():
        for use_threat in (False, True):
            label = f'{scen_name} | Technique3={"ON" if use_threat else "OFF"}'
            factory = (lambda ut=use_threat: ThreatAwareAgent(use_threat=ut))
            res = experiment(factory, pool, repeat_nums=REPEATS, label=label)
            results.append(res)

    print('\n===== Technique 3 Ablation Results =====')
    print(f"{'Condition':<55}{'AvgSC':>8}{'StdSC':>8}{'Win%':>8}{'Surv%':>8}{'Def%':>8}")
    for r in results:
        print(f"{r['label']:<55}{r['avg_sc']:>8}{r['std_sc']:>8}{r['win_rate']:>8}{r['survive_rate']:>8}{r['defeat_rate']:>8}")
