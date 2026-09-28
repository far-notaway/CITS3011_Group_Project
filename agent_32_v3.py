import time
#import timeout_decorator
'''
WINDOWS COMPATIBILITY NOTE:
    The timeout_decorator package may not work correctly on Windows. For local
    development on Windows, you may comment out the import and all four
    @timeout_decorator.timeout(1) lines in this file. If you do so, measure the
    running time of __init__, new_game, update_game, and get_actions yourself
    (for example, with time.perf_counter). This local workaround does not relax
    the one-second limit: it is a hard constraint and will be enforced
    independently during marking.
'''
from collections import defaultdict, deque
import random
import networkx as nx
from agent_baselines import Agent

# Map topology (and the distances derived from it) never changes between games, so both are cached at
# module level, keyed by map name, instead of being rebuilt every new_game() call across all test games.
_MAP_GRAPH_CACHE = {}
_MAP_DISTANCE_CACHE = {}


def unit_loc(unit):
    '''Extracts the 3-letter province code from a unit string, handling the leading * of dislodged units.'''
    unit = unit[1:] if unit.startswith('*') else unit
    return unit[2:5]


def _base(loc):
    '''Strips coastal qualifiers (e.g. STP/NC -> STP) so multi-coast provinces map to a single graph node.'''
    return loc.split('/')[0].upper()


def _bfs(graph, source):
    '''Breadth-first distances from source to every reachable node. All edges are unweighted (each hop
    costs 1), so BFS gives the same result as Dijkstra here with less overhead.'''
    dist = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in dist:
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)
    return dist


class StudentAgent(Agent):
    '''
    Basic technique: a heuristic 1-ply evaluation (distance-to-nearest enemy supply
    centre + capture value) used to greedily pick a move/hold for every unit, in the
    same spirit as the GreedyAgent baseline but with a richer scoring function.

    Three new techniques (each toggleable independently for isolated experiments):
      - use_defense:        defensive awareness. Threatened home centres are held
                             (or reinforced) instead of being greedily abandoned to attack.
      - use_support_coord:  support coordination. Units whose greedy targets overlap,
                             or whose own attack is low-value, are converted into support
                             orders for the strongest unit/hold instead of acting alone.
      - use_opponent_model: tit-for-tat opponent modelling. Each opponent starts FRIENDLY
                             (cooperate first). Each movement phase we classify their last
                             action toward us - an attack makes them HOSTILE (retaliate),
                             a support makes them FRIENDLY (always forgiven), anything else
                             leaves their standing unchanged. Each power also has its own
                             forgiveness probability (a random chance to forgive a HOSTILE
                             power anyway) that rises 5% whenever they play nice and drops
                             7% whenever they play hostile, so a repeat offender gradually
                             stops getting the benefit of the doubt while a power that only
                             slipped up once is quickly given another chance.
      - use_terminal_stab:  the "Terminal Stab" endgame override. GTFT assumes trust is worth
                             preserving for future turns, but Diplomacy ends the instant someone
                             reaches 18 centres. Once we hold terminal_stab_threshold centres AND
                             enough enemy/neutral centres lie within terminal_stab_horizon hops of
                             our units to plausibly finish the game, retaliation stops mattering -
                             we permanently abandon tit-for-tat and attack every power at full
                             aggression, FRIENDLY standing included, to race for the win.
    '''

    #@timeout_decorator.timeout(1)
    def __init__(self, agent_name='Group32 Agent', use_defense=True, use_support_coord=True,
                 use_opponent_model=True, forgiveness_prob=0.1,
                 forgive_increment=0.0, forgive_decrement=0.0,
                 use_terminal_stab=True, terminal_stab_threshold=16,
                 terminal_stab_horizon=3, terminal_stab_win_centers=18):
        super().__init__(agent_name)
        self.use_defense = use_defense
        self.use_support_coord = use_support_coord
        self.use_opponent_model = use_opponent_model
        self.initial_forgiveness_prob = forgiveness_prob
        self.forgive_increment = forgive_increment  # added to forgiveness_prob when a power plays nice
        self.forgive_decrement = forgive_decrement  # subtracted from forgiveness_prob when a power plays hostile
        self.forgiveness_prob = {}
        self.use_terminal_stab = use_terminal_stab
        self.terminal_stab_threshold = terminal_stab_threshold  # own centres needed before a stab is even considered
        self.terminal_stab_horizon = terminal_stab_horizon  # hops within which a centre counts as reachable in time
        self.terminal_stab_win_centers = terminal_stab_win_centers
        self.terminal_stab_active = False  # once tripped, stays on for the rest of the game
        self.map_graph_army = None
        self.map_graph_navy = None
        self.attitude = {}

    #@timeout_decorator.timeout(1)
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name
        self._build_map_graphs()
        # Cooperate on the first move: every opponent starts in our good books.
        self.attitude = {p: 'FRIENDLY' for p in self.game.powers.keys() if p != power_name}
        self.forgiveness_prob = {p: self.initial_forgiveness_prob for p in self.attitude}
        self.terminal_stab_active = False

    def _build_map_graphs(self):
        '''Connection graph of the map (province adjacency), separately for armies and fleets, plus
        BFS distance from every supply centre. Both are cached at module level keyed by map name since
        the map topology is fixed, so this only does real work once no matter how many agents/games run.'''
        map_name = self.game.map.name
        if map_name not in _MAP_GRAPH_CACHE:
            army_graph = nx.Graph()
            navy_graph = nx.Graph()

            raw_locations = list(self.game.map.loc_type.keys())
            for loc in raw_locations:
                base = _base(loc)
                if self.game.map.loc_type[loc] in ('LAND', 'COAST'):
                    army_graph.add_node(base)
                if self.game.map.loc_type[loc] in ('WATER', 'COAST'):
                    navy_graph.add_node(base)

            upper_locations = [loc.upper() for loc in raw_locations]
            for i in upper_locations:
                for j in upper_locations:
                    if self.game.map.abuts('A', i, '-', j):
                        army_graph.add_edge(_base(i), _base(j))
                    if self.game.map.abuts('F', i, '-', j):
                        navy_graph.add_edge(_base(i), _base(j))

            # Distance from every supply centre to every reachable node, computed once via BFS and reused
            # every turn: per-move distance-to-nearest-enemy-centre then just mins over these fixed lists.
            centers = [_base(c) for c in self.game.map.scs]
            sc_dist_army = {c: _bfs(army_graph, c) for c in centers if c in army_graph}
            sc_dist_navy = {c: _bfs(navy_graph, c) for c in centers if c in navy_graph}

            _MAP_GRAPH_CACHE[map_name] = (army_graph, navy_graph)
            _MAP_DISTANCE_CACHE[map_name] = (sc_dist_army, sc_dist_navy)

        self.map_graph_army, self.map_graph_navy = _MAP_GRAPH_CACHE[map_name]
        self._sc_dist_army, self._sc_dist_navy = _MAP_DISTANCE_CACHE[map_name]

    def _provinces_adjacent(self, loc_a, loc_b):
        return self.game.map.abuts('A', loc_a, '-', loc_b) or self.game.map.abuts('F', loc_a, '-', loc_b)

    #@timeout_decorator.timeout(1) # This is only for updating the game engine and other states if any. Do not implement heavy stratergy here.
    def update_game(self, all_power_orders):
        if self.use_opponent_model and self.game.phase_type == 'M':
            self._update_opponent_model(all_power_orders)

        # do not make changes to the following codes
        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()

    def _update_opponent_model(self, all_power_orders):
        '''Tit-for-tat: mirror each opponent's last action toward us, with random forgiveness.'''
        self_units = set(self.game.get_units(self.power_name))
        self_locs = set(self.game.get_centers(self.power_name)) | {unit_loc(u) for u in self_units}

        for power_name, orders in all_power_orders.items():
            if power_name not in self.attitude:
                continue
            hostile, friendly = False, False
            for order in orders:
                words = order.split(' ')
                if len(words) < 3:
                    continue
                if words[2] == '-' and len(words) >= 4 and words[3] in self_locs:
                    hostile = True  # moved/attacked into our territory
                elif words[2] == 'S' and len(words) == 5:
                    if f'{words[3]} {words[4]}' in self_units:
                        friendly = True  # supported one of our units to hold
                elif words[2] == 'S' and len(words) == 7 and words[5] == '-':
                    if f'{words[3]} {words[4]}' in self_units:
                        friendly = True  # supported one of our units to move
                    elif words[6] in self_locs:
                        hostile = True  # supported someone else's attack on us
                elif words[2] == 'C' and len(words) == 7 and words[5] == '-' and words[6] in self_locs:
                    hostile = True  # convoyed an army that is landing an attack on us

            if friendly:
                self.attitude[power_name] = 'FRIENDLY'  # a helping hand is forgiven immediately
                self.forgiveness_prob[power_name] = min(1.0, self.forgiveness_prob[power_name] + 0.03) #self.forgive_increment
            elif hostile:
                if random.random() < self.forgiveness_prob[power_name]:
                    self.attitude[power_name] = 'NEUTRAL'  # generous tit-for-tat: forgive sometimes
                else:
                    self.attitude[power_name] = 'HOSTILE'  # retaliate in kind
                self.forgiveness_prob[power_name] = max(0.0, self.forgiveness_prob[power_name] - 0.1) #self.forgive_decrement
            # a neutral turn (hold, convoy, or action elsewhere) leaves the standing unchanged

    def _opponent_weights(self):
        '''Higher weight => more attractive attack target: retaliate against HOSTILE powers,
        go easy on FRIENDLY ones, and still lean slightly toward weaker powers either way.
        Once a terminal stab is triggered, GTFT is dropped entirely: every power is weighted as
        a full-aggression target regardless of standing, since the game ends before anyone can
        coordinate retaliation.'''
        all_centers = self.game.get_centers()
        opp_centers = {p: len(all_centers.get(p, [])) for p in self.attitude}
        avg_centers = max(1.0, sum(opp_centers.values()) / max(1, len(opp_centers)))
        if self.terminal_stab_active:
            attitude_weight = defaultdict(lambda: 2.0)
        else:
            attitude_weight = {'HOSTILE': 1.8, 'NEUTRAL': 1.0, 'FRIENDLY': 0.5}

        weights = {}
        for p, att in self.attitude.items():
            weakness_bonus = max(-0.2, min(0.3, (avg_centers - opp_centers.get(p, 0)) / avg_centers))
            weights[p] = attitude_weight[att] + weakness_bonus
        return weights

    def _update_terminal_stab(self):
        '''Trip the "Terminal Stab" override once we hold enough centres that a fast final assault
        could plausibly reach the 18-centre win before opponents get a chance to counter-attack.
        This is a one-way switch: once retaliation stops mattering it never starts mattering again.'''
        if not self.use_terminal_stab or self.terminal_stab_active:
            return
        my_centers = len(self.game.get_centers(self.power_name))
        if my_centers < self.terminal_stab_threshold:
            return
        centers_needed = self.terminal_stab_win_centers - my_centers
        if centers_needed <= 0:
            self.terminal_stab_active = True
            return
        if self._reachable_center_count(self.terminal_stab_horizon) >= centers_needed:
            self.terminal_stab_active = True

    def _reachable_center_count(self, horizon):
        '''Number of enemy/neutral supply centres within `horizon` hops of any of our own units,
        reusing the module-cached per-centre BFS distances so this is a cheap lookup, not a new search.'''
        my_units = self.game.get_units(self.power_name)
        my_centers = set(self.game.get_centers(self.power_name))
        unit_locs_by_type = defaultdict(set)
        for u in my_units:
            unit_locs_by_type[u[0]].add(unit_loc(u))

        count = 0
        for center in self.game.map.scs:
            if center in my_centers:
                continue
            best = None
            for unit_type, locs in unit_locs_by_type.items():
                sc_dist = self._sc_dist_army if unit_type == 'A' else self._sc_dist_navy
                dists = sc_dist.get(center, {})
                for loc in locs:
                    dist = dists.get(loc)
                    if dist is not None and (best is None or dist < best):
                        best = dist
            if best is not None and best <= horizon:
                count += 1
        return count

    #@timeout_decorator.timeout(1)
    def get_actions(self):
        phase_type = self.game.phase_type
        if phase_type == 'M':
            return self._movement_actions()
        if phase_type == 'R':
            return self._retreat_actions()
        return self._adjustment_actions()

    # ------------------------------------------------------------------
    # Movement phase
    # ------------------------------------------------------------------

    def _movement_actions(self):
        possible_orders = self.game.get_all_possible_orders()
        orderable_locations = self.game.get_orderable_locations(self.power_name)
        if not orderable_locations:
            return []

        centers_by_power = self.game.get_centers()
        units_by_power = self.game.get_units()
        my_centers = set(centers_by_power.get(self.power_name, []))

        loc_owner = {}
        for p, cs in centers_by_power.items():
            for c in cs:
                loc_owner[c] = p

        unit_owner = {}
        for p, units in units_by_power.items():
            for u in units:
                unit_owner[unit_loc(u)] = p

        enemy_centers = [c for c in self.game.map.scs if c not in my_centers]
        self._enemy_centers = set(enemy_centers)
        if self.use_terminal_stab:
            self._update_terminal_stab()
        opp_weights = self._opponent_weights() if self.use_opponent_model or self.terminal_stab_active else {}

        # distance (in hops) from every province to its nearest enemy supply centre: combined on the fly
        # from the module-cached per-supply-centre BFS distances, so no graph traversal happens per turn.
        self._dist_army = self._nearest_enemy_distance(self._sc_dist_army, enemy_centers)
        self._dist_navy = self._nearest_enemy_distance(self._sc_dist_navy, enemy_centers)

        occupied_enemy_locs = {loc for loc, p in unit_owner.items() if p != self.power_name}

        # A centre is only genuinely at risk if at least 2 enemy units could combine against it - a lone
        # attacker moving into a held province just bounces, so a single neighbour is not a real threat.
        # Units belonging to a currently FRIENDLY power (tit-for-tat) are trusted and excluded.
        threatened = {}
        if self.use_defense:
            threatening_locs = {loc for loc in occupied_enemy_locs
                                 if self.attitude.get(unit_owner[loc]) != 'FRIENDLY'} if self.use_opponent_model else occupied_enemy_locs
            for c in my_centers:
                threat = sum(1 for loc in threatening_locs if self._provinces_adjacent(loc, c))
                if threat >= 2:
                    threatened[c] = threat

        # Every unit's move/hold options, best first.
        unit_candidates = {}
        for loc in orderable_locations:
            orders = possible_orders.get(loc, [])
            if not orders:
                continue
            unit_type = orders[0][0]
            options = []
            for order in orders:
                words = order.split(' ')
                if words[2] == 'H':
                    options.append((self._score_hold(loc, threatened), 'H', loc, order))
                elif words[2] == '-' and 'VIA' not in words:
                    target = _base(words[3])
                    score = self._score_move(unit_type, target, loc_owner, enemy_centers, opp_weights)
                    options.append((score, 'M', target, order))
                elif words[2] == '-' and 'VIA' in words:
                    # convoy move: scored the same as a normal move, but only kept if
                    # _apply_convoy_coordination can find a fleet chain we fully control
                    target = _base(words[3])
                    score = self._score_move(unit_type, target, loc_owner, enemy_centers, opp_weights)
                    options.append((score, 'VIA', target, order))
                # support/convoy orders are decided in the coordination step below
            if options:
                options.sort(key=lambda x: x[0], reverse=True)
                unit_candidates[loc] = {'unit_type': unit_type, 'options': options}

        # Greedily assign each unit its best available target. Empty/uncontested targets can only be
        # claimed by one unit each so units spread out to grab distinct centres instead of bunching up
        # on the same square (which would just waste every extra unit's turn); targets already occupied
        # by an enemy unit can be shared, since combining several attackers there is a real benefit.
        claimed = {}
        candidates = {}
        for loc in sorted(unit_candidates, key=lambda l: unit_candidates[l]['options'][0][0], reverse=True):
            info = unit_candidates[loc]
            chosen = info['options'][-1]
            for score, kind, target, order in info['options']:
                if kind == 'H' or target in occupied_enemy_locs or target not in claimed:
                    chosen = (score, kind, target, order)
                    if kind in ('M', 'VIA') and target not in occupied_enemy_locs:
                        claimed[target] = loc
                    break
            score, kind, target, order = chosen
            candidates[loc] = {'order': order, 'score': score, 'kind': kind, 'target': target,
                                'unit_type': info['unit_type']}

        if self.use_defense:
            self._apply_defense(candidates, possible_orders, threatened)

        self._apply_convoy_coordination(candidates, unit_candidates)

        if self.use_support_coord:
            self._apply_support_coordination(candidates, possible_orders, occupied_enemy_locs, unit_candidates)

        return [cand['order'] for cand in candidates.values()]

    @staticmethod
    def _nearest_enemy_distance(sc_dist, enemy_centers):
        '''Combine the module-cached per-supply-centre BFS distances into a per-node distance to the
        nearest currently-enemy centre, without re-traversing the graph.'''
        nearest = {}
        for center in enemy_centers:
            for node, dist in sc_dist.get(center, {}).items():
                if node not in nearest or dist < nearest[node]:
                    nearest[node] = dist
        return nearest

    def _score_hold(self, loc, threatened):
        score = 0.5
        if loc in threatened:
            score += 1.0 + 0.3 * threatened[loc]
        elif loc in self._enemy_centers:
            # a centre we just moved onto is not ours yet - ownership only locks in at the next
            # Winter adjustment, so holding here is worth far more than wandering into open territory
            score += 2.5
        return score

    def _score_move(self, unit_type, target, loc_owner, enemy_centers, opp_weights):
        score = 0.0
        owner = loc_owner.get(target)
        if target in enemy_centers:
            score += 3.0 * opp_weights.get(owner, 1.0) if owner else 3.0
        elif owner == self.power_name:
            score += 0.2  # repositioning within own territory
        else:
            score += 0.3  # expanding into open/neutral non-centre territory

        dist_map = self._dist_army if unit_type == 'A' else self._dist_navy
        dist = dist_map.get(target)
        if dist is not None:
            score += max(0.0, 2.0 - 0.3 * dist)
        return score

    def _apply_defense(self, candidates, possible_orders, threatened):
        '''Defensive awareness: keep/reinforce threatened home centres instead of abandoning them.'''
        my_unit_locs = set(candidates.keys())

        # 1. Prefer holding a threatened centre over a move that isn't clearly more valuable.
        for loc in list(candidates.keys()):
            if loc in threatened and candidates[loc]['kind'] == 'M':
                hold_score = self._score_hold(loc, threatened)
                if candidates[loc]['score'] < hold_score + 0.75:
                    hold_order = next((o for o in possible_orders[loc] if o.split(' ')[2] == 'H'), None)
                    if hold_order:
                        candidates[loc] = {'order': hold_order, 'score': hold_score, 'kind': 'H',
                                            'target': loc, 'unit_type': candidates[loc]['unit_type']}

        # 2. Garrison threatened centres that currently have no unit of ours, by redirecting a neighbour.
        undefended = [c for c in threatened if c not in my_unit_locs]
        for center in undefended:
            best_loc = None
            for loc, cand in candidates.items():
                if cand['kind'] != 'M':
                    continue
                order = f"{cand['unit_type']} {loc} - {center}"
                if order in possible_orders.get(loc, []):
                    if best_loc is None or cand['score'] < candidates[best_loc]['score']:
                        best_loc = loc
            if best_loc:
                candidates[best_loc] = {'order': f"{candidates[best_loc]['unit_type']} {best_loc} - {center}",
                                         'score': self._score_hold(center, threatened), 'kind': 'M',
                                         'target': center, 'unit_type': candidates[best_loc]['unit_type']}

    def _apply_convoy_coordination(self, candidates, unit_candidates):
        '''Convoy coordination: an island power (or an army with no land route) can only ever act on a
        'VIA' move if every fleet along a connecting sea path is ALSO ordered to convoy it this same
        turn - Diplomacy does not let a lone army order itself onto a boat. game.convoy_paths_dest
        (built as a side effect of get_all_possible_orders) lists, for every source/destination pair,
        each complete set of fleets that forms a valid path, regardless of which power owns them. We
        only ever pick a path made entirely of our own fleets (we cannot order an enemy fleet to help),
        preferring the shortest one, and never reuse a fleet already committed to a different convoy
        this turn. If no such path exists, the VIA move is undoable this turn, so we fall back to that
        army's next-best option instead of submitting an order that is guaranteed to fail.'''
        own_fleet_locs = {loc for loc, info in unit_candidates.items() if info['unit_type'] == 'F'}
        via_locs = [loc for loc, cand in candidates.items() if cand['kind'] == 'VIA']
        via_locs.sort(key=lambda l: candidates[l]['score'], reverse=True)

        committed_fleets = set()
        for army_loc in via_locs:
            dest = candidates[army_loc]['target']
            paths = self.game.convoy_paths_dest.get(army_loc, {}).get(dest, [])
            usable_paths = [path for path in paths
                             if path.issubset(own_fleet_locs) and not (path & committed_fleets)]
            if usable_paths:
                path = min(usable_paths, key=len)
                committed_fleets |= set(path)
                for fleet_loc in path:
                    candidates[fleet_loc] = {'order': f"F {fleet_loc} C A {army_loc} - {dest}",
                                              'score': candidates[fleet_loc]['score'] + 0.5,
                                              'kind': 'C', 'target': army_loc, 'unit_type': 'F'}
                continue

            for score, kind, target, order in unit_candidates[army_loc]['options']:
                if kind == 'VIA':
                    continue
                candidates[army_loc] = {'order': order, 'score': score, 'kind': kind,
                                         'target': target, 'unit_type': 'A'}
                break

    def _apply_support_coordination(self, candidates, possible_orders, occupied_enemy_locs, unit_candidates):
        '''Support coordination: turn overlapping/low-value single-unit actions into joint attack/hold combos.'''
        target_groups = defaultdict(list)
        for loc, cand in candidates.items():
            if cand['kind'] == 'C':
                continue  # already committed to convoying an army this turn - leave it alone
            target_groups[cand['target']].append(loc)

        for target, locs in target_groups.items():
            if len(locs) < 2:
                continue
            primary_loc = max(locs, key=lambda l: candidates[l]['score'])
            primary = candidates[primary_loc]
            for loc in locs:
                if loc == primary_loc:
                    continue
                cand = candidates[loc]
                if primary['kind'] == 'H':
                    support_order = f"{cand['unit_type']} {loc} S {primary['unit_type']} {target}"
                else:
                    support_order = f"{cand['unit_type']} {loc} S {primary['unit_type']} {primary_loc} - {target}"
                if support_order in possible_orders.get(loc, []):
                    candidates[loc] = {'order': support_order, 'score': cand['score'] + 0.5, 'kind': 'S',
                                        'target': target, 'unit_type': cand['unit_type']}

        # Low-value units support important holds (typically defensive holds) instead of acting alone.
        for held_loc, held in list(candidates.items()):
            if held['kind'] != 'H':
                continue
            for loc, cand in candidates.items():
                if loc == held_loc or cand['kind'] in ('S', 'C') or cand['score'] >= 1.5:
                    continue
                support_order = f"{cand['unit_type']} {loc} S {held['unit_type']} {held_loc}"
                if support_order in possible_orders.get(loc, []):
                    candidates[loc] = {'order': support_order, 'score': cand['score'] + 0.5, 'kind': 'S',
                                        'target': held_loc, 'unit_type': cand['unit_type']}

        # A lone attacker moving into an enemy-occupied centre just bounces off a unit that holds with
        # no support, so recruit a spare neighbour to support the attack and guarantee it. If nobody can
        # help this turn, the attack would fail for free, so fall back to the attacker's next-best option
        # instead of wasting the whole game bouncing off the same defended centre.
        already_supported_targets = {cand['target'] for cand in candidates.values() if cand['kind'] == 'S'}

        attackers = defaultdict(list)
        for loc, cand in candidates.items():
            if cand['kind'] == 'M' and cand['target'] in occupied_enemy_locs:
                attackers[cand['target']].append(loc)

        used_targets = {cand['target'] for cand in candidates.values()}

        for target, attacker_locs in attackers.items():
            if len(attacker_locs) != 1 or target in already_supported_targets:
                continue  # already coordinated above, or nobody is attacking it
            attacker_loc = attacker_locs[0]
            attacker = candidates[attacker_loc]
            helper_loc = None
            for loc, cand in candidates.items():
                if loc == attacker_loc or cand['kind'] in ('S', 'C'):
                    continue
                if cand['target'] in occupied_enemy_locs and cand['target'] != target:
                    continue  # do not cannibalise another live attack
                support_order = f"{cand['unit_type']} {loc} S {attacker['unit_type']} {attacker_loc} - {target}"
                if support_order in possible_orders.get(loc, []):
                    if helper_loc is None or cand['score'] < candidates[helper_loc]['score']:
                        helper_loc = loc
            if helper_loc:
                cand = candidates[helper_loc]
                candidates[helper_loc] = {
                    'order': f"{cand['unit_type']} {helper_loc} S {attacker['unit_type']} {attacker_loc} - {target}",
                    'score': cand['score'] + 0.5, 'kind': 'S', 'target': target, 'unit_type': cand['unit_type']}
            else:
                for score, kind, alt_target, order in unit_candidates[attacker_loc]['options']:
                    if alt_target == target:
                        continue  # this is exactly the unsupported attack that just failed to find help
                    if kind == 'H' or alt_target in occupied_enemy_locs or alt_target not in used_targets:
                        candidates[attacker_loc] = {'order': order, 'score': score, 'kind': kind,
                                                     'target': alt_target, 'unit_type': attacker['unit_type']}
                        used_targets.add(alt_target)
                        break

    # ------------------------------------------------------------------
    # Retreat phase
    # ------------------------------------------------------------------

    def _retreat_actions(self):
        possible_orders = self.game.get_all_possible_orders()
        orderable_locations = self.game.get_orderable_locations(self.power_name)
        if not orderable_locations:
            return []

        my_centers = set(self.game.get_centers(self.power_name))
        units_by_power = self.game.get_units()
        enemy_unit_locs = [unit_loc(u) for p, units in units_by_power.items() if p != self.power_name for u in units]

        orders = []
        for loc in orderable_locations:
            options = possible_orders.get(loc, [])
            if not options:
                continue
            best_order, best_score = None, None
            for order in options:
                words = order.split(' ')
                if words[2] == 'D':
                    score = -1.0  # disbanding is a last resort
                else:
                    dest = words[3]
                    score = 1.0
                    if dest in my_centers:
                        score += 2.0
                    score -= 0.5 * sum(1 for loc2 in enemy_unit_locs if self._provinces_adjacent(loc2, dest))
                if best_score is None or score > best_score:
                    best_score, best_order = score, order
            orders.append(best_order)
        return orders

    # ------------------------------------------------------------------
    # Adjustment (build/disband) phase
    # ------------------------------------------------------------------

    def _adjustment_actions(self):
        possible_orders = self.game.get_all_possible_orders()
        orderable_locations = self.game.get_orderable_locations(self.power_name)
        if not orderable_locations:
            return []

        my_centers = set(self.game.get_centers(self.power_name))
        my_units = self.game.get_units(self.power_name)
        build_count = len(my_centers) - len(my_units)

        units_by_power = self.game.get_units()
        enemy_unit_locs = [unit_loc(u) for p, units in units_by_power.items() if p != self.power_name for u in units]

        if build_count > 0:
            orders = []
            for loc in orderable_locations:
                options = [o for o in possible_orders.get(loc, []) if o != 'WAIVE']
                if not options:
                    continue
                army_order = next((o for o in options if o.startswith('A ')), None)
                navy_order = next((o for o in options if o.startswith('F ')), None)
                if army_order and navy_order:
                    # Build a fleet at coastal sites under naval threat, an army otherwise.
                    threat = sum(1 for loc2 in enemy_unit_locs if self._provinces_adjacent(loc2, loc))
                    orders.append(navy_order if threat and loc in self.map_graph_navy else army_order)
                else:
                    orders.append(army_order or navy_order)
            return orders

        if build_count < 0:
            need = -build_count
            scored_units = []
            for u in my_units:
                loc = unit_loc(u)
                threat = sum(1 for loc2 in enemy_unit_locs if self._provinces_adjacent(loc2, loc))
                value = 2.0 if loc in my_centers else 0.5
                scored_units.append((value + 0.5 * threat, u))
            scored_units.sort(key=lambda x: x[0])  # least valuable / least threatened units disbanded first
            return [f'{u} D' for _, u in scored_units[:need]]

        return []
