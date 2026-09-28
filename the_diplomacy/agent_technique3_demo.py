"""
Standalone demo agent for New Technique 3 (Adaptive Threat Modelling).

This is NOT the final submission agent - it exists so Technique 3 can be
developed, run, and evaluated independently of whatever Nam (basic +
technique 1) and Nathaniel build. It uses a minimal
greedy-toward-nearest-target fallback for movement (deliberately simple,
NOT the final basic technique - just enough scaffolding to see Technique 3
change behaviour), so the ablation comparison below isolates the effect of
threat-awareness rather than mixing it with someone else's improvements.

Once the team's pieces exist, ThreatModel.get_normalised_threats(...) is
meant to be called from wherever target-scoring happens (Nam's
assignment step) and wherever defensive unit allocation happens (Nathaniel's
CSP step) - see the integration note at the bottom of this file.

Usage:
    from agent_technique3_demo import ThreatAwareAgent
    agent = ThreatAwareAgent(use_threat=True)   # technique 3 ON
    agent = ThreatAwareAgent(use_threat=False)  # technique 3 OFF (ablation)
"""

import random
import platform
import networkx as nx

from agent_baselines import Agent
from technique3_threat import ThreatModel

# timeout_decorator relies on signal.SIGALRM, which does not exist on
# Windows (see the project README's Windows compatibility note). For local
# dev on Windows we fall back to a no-op decorator; the 1s limit is still a
# hard constraint and is enforced independently at marking time on Linux.
# If you're on Windows and want to sanity-check timing yourself, wrap calls
# with time.perf_counter() instead (see technique3_threat.py's docstring).
if platform.system() == 'Windows':
    def timeout(seconds):
        def decorator(func):
            return func
        return decorator
else:
    import timeout_decorator
    timeout = timeout_decorator.timeout


class ThreatAwareAgent(Agent):

    def __init__(self, agent_name='Threat-Aware Agent', use_threat=True):
        super().__init__(agent_name)
        self.use_threat = use_threat
        self.threat_model = ThreatModel() if use_threat else None
        self.map_graph_army = None
        self.map_graph_navy = None

    @timeout(1)
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name
        self._build_map_graphs()
        if self.use_threat:
            self.threat_model.init_game(game, power_name)

    def _build_map_graphs(self):
        self.map_graph_army = nx.Graph()
        self.map_graph_navy = nx.Graph()
        locations = list(self.game.map.loc_type.keys())
        for loc in locations:
            if self.game.map.loc_type[loc] in ('LAND', 'COAST'):
                self.map_graph_army.add_node(loc.upper())
            if self.game.map.loc_type[loc] in ('WATER', 'COAST'):
                self.map_graph_navy.add_node(loc.upper())
        locations = [loc.upper() for loc in locations]
        for i in locations:
            for j in locations:
                if self.game.map.abuts('A', i, '-', j):
                    self.map_graph_army.add_edge(i, j)
                if self.game.map.abuts('F', i, '-', j):
                    self.map_graph_navy.add_edge(i, j)

    @timeout(1)
    def update_game(self, all_power_orders):
        if self.use_threat and self.game.phase_type == 'M':
            self_centers_before = self.game.get_centers(self.power_name)
            self_unit_locs_before = self.game.get_orderable_locations(self.power_name)
            self_locs_before = list(set(self_centers_before + self_unit_locs_before))
            self_units_before = self.game.get_units(power_name=self.power_name)
            do_update = True
        else:
            do_update = False

        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()

        if do_update:
            self.threat_model.update_after_movement(
                self.game, all_power_orders, self_locs_before, self_units_before
            )

    @timeout(1)
    def get_actions(self):
        all_possible_orders = self.game.get_all_possible_orders()
        orderable_locations = self.game.get_orderable_locations(self.power_name)

        if self.game.phase_type != 'M':
            return [random.choice(all_possible_orders[loc])
                    for loc in orderable_locations if all_possible_orders[loc]]

        # --- who owns each contested location right now ---
        owner_of_loc = {}
        for p in self.game.powers.keys():
            for loc in set(self.game.get_centers(p) + self.game.get_orderable_locations(p)):
                owner_of_loc[loc] = p

        threats = self.threat_model.get_normalised_threats(self.game) if self.use_threat else {}

        enemy_centers = [c for c in self.game.map.scs
                          if c not in self.game.get_centers(self.power_name)]

        power_orders = []
        for loc in orderable_locations:
            army_orders = [o for o in all_possible_orders[loc] if o[0] == 'A']
            navy_orders = [o for o in all_possible_orders[loc] if o[0] == 'F']

            if army_orders:
                order = self._choose_move(loc, army_orders, enemy_centers,
                                           self.map_graph_army, owner_of_loc, threats)
                if order:
                    power_orders.append(order)
            if navy_orders:
                order = self._choose_move(loc, navy_orders, enemy_centers,
                                           self.map_graph_navy, owner_of_loc, threats)
                if order:
                    power_orders.append(order)

            if not army_orders and not navy_orders and all_possible_orders[loc]:
                power_orders.append(random.choice(all_possible_orders[loc]))

        return list(set(power_orders))

    def _choose_move(self, loc, possible_orders, enemy_centers, graph, owner_of_loc, threats):
        """Pick nearest enemy centre, but with Technique 3 ON, discount
        centres owned/defended by a currently high-threat power (they're
        likely to be strongly defended / retaliate) and prefer holding at
        centres under our control that face border pressure from a
        high-threat neighbour."""
        if loc not in graph.nodes:
            return random.choice(possible_orders) if possible_orders else None

        if loc in enemy_centers:
            unit_type = possible_orders[0][0]
            return f'{unit_type} {loc} H'

        try:
            paths = nx.shortest_path(graph, source=loc)
        except Exception:
            return random.choice(possible_orders) if possible_orders else None

        best_target, best_score = None, -1e18
        for center in enemy_centers:
            if center not in paths:
                continue
            dist = len(paths[center])
            base_score = 10.0 / dist

            if self.use_threat:
                owner = owner_of_loc.get(center)
                threat = threats.get(owner, 0.0) if owner else 0.0
                # High-threat owner -> discount this target (likely
                # defended / will retaliate). Unowned/no-threat -> no penalty.
                base_score -= 4.0 * threat

            if base_score > best_score:
                best_score, best_target = base_score, center

        if best_target is None or best_target not in paths or len(paths[best_target]) <= 1:
            return random.choice(possible_orders) if possible_orders else None

        next_hop = paths[best_target][1]
        unit_type = possible_orders[0][0]
        candidate = f'{unit_type} {loc} - {next_hop}'
        return candidate if candidate in possible_orders else random.choice(possible_orders)


# ---------------------------------------------------------------------
# Integration note for merging into the final agent_groupnumber.py:
#
#   1. Copy the ThreatModel class from technique3_threat.py in as-is.
#   2. In StudentAgent.new_game, call self.threat_model.init_game(...).
#   3. In StudentAgent.update_game, call
#      self.threat_model.update_after_movement(...) right after
#      game.process() on a movement phase (capture self_locs/self_units
#      BEFORE process(), same as this file does).
#   4. In get_actions, call self.threat_model.get_normalised_threats(...)
#      once per turn and pass the resulting dict into Person A's target
#      scoring function and/or Person B's support-coordination step as an
#      extra weighting term.
# ---------------------------------------------------------------------
