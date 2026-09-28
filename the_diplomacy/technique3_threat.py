"""
New Technique 3: Adaptive Opponent Threat Modelling
=====================================================

Motivation
----------
The provided AttitudeAgent models opponents with THREE discrete buckets
(FRIENDLY / NEUTRAL / HOSTILE) that jump around a fixed transition-probability
table. That is a reasonable baseline, but it throws away information:

  1. It cannot distinguish "mildly annoying neighbour" from "all-out attacker" -
     both just become HOSTILE.
  2. It ignores *proximity threat* - an enemy stack of 4 units sitting on your
     border is dangerous even before it has fired a shot, but AttitudeAgent
     only reacts after an attack/support event happens.
  3. It has no decay - once a power is judged HOSTILE from one attack, the
     label persists until another qualifying event resets it, even if that
     power hasn't done anything since.

This module replaces the 3-bucket label with a **continuous, decaying threat
score per opponent power**, built from two independent signals:

  (a) Behavioural signal - attacks / supports directed at us or against us,
      accumulated with an exponential moving average (EMA) so recent actions
      matter more than old ones, and one quiet turn lets threat cool down
      rather than requiring an explicit "friendly" event to reset it.

  (b) Structural/positional signal - how many enemy units are adjacent to our
      centres and units right now, using the same map-graph idea as
      GreedyAgent's build_map_graphs, so a build-up we can *see* on the map
      counts even if that power has not attacked yet.

The combined, normalised score is then usable as a weight anywhere a "how
dangerous is power X to me" number is needed: target scoring, defensive unit
allocation, or (later) as an input feature into technique 1's assignment
cost matrix / technique 2's CSP support decisions.

This file is self-contained and can be dropped into agent_groupnumber.py by
copy-pasting the class body into StudentAgent (or composed by delegation, as
the demo agent in agent_technique3_demo.py does).
"""

import networkx as nx


class ThreatModel:
    """Tracks a continuous, decaying threat score for every opponent power.

    threat_scores are floats, nominally in roughly [-3, 5] before
    normalisation (see get_normalised_threats), where:
        negative -> power has been actively helping us / no pressure
        ~0       -> neutral / unknown
        positive -> power has been attacking us and/or massing on our border
    """

    def __init__(self, decay=0.65, border_weight=0.35, max_abs_event=3.0):
        """
        decay:          how much of the *old* EMA score is kept each update
                         (0 = fully reactive to last turn only, 1 = never
                         updates). 0.65 means last turn's events still fade
                         out over a handful of turns rather than lingering
                         for the whole game like a fixed HOSTILE label would.
        border_weight:   how much the positional (border pressure) term
                         contributes relative to the behavioural EMA term.
        max_abs_event:   clip a single turn's behavioural event score to
                         +/- this value so one big multi-unit attack turn
                         doesn't blow the EMA out to an extreme that takes
                         forever to recover from.
        """
        self.decay = decay
        self.border_weight = border_weight
        self.max_abs_event = max_abs_event
        self.behaviour_scores = {}   # power_name -> EMA float
        self.map_graph_army = None
        self.map_graph_navy = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def init_game(self, game, power_name):
        self.power_name = power_name
        self.behaviour_scores = {
            p: 0.0 for p in game.powers.keys() if p != power_name
        }
        self._build_map_graphs(game)

    def _build_map_graphs(self, game):
        """Same idea as GreedyAgent.build_map_graphs - the connectivity
        graph of the map (not the game's state-space), used to measure how
        close enemy units are to our territory."""
        self.map_graph_army = nx.Graph()
        self.map_graph_navy = nx.Graph()

        locations = list(game.map.loc_type.keys())
        for loc in locations:
            if game.map.loc_type[loc] in ('LAND', 'COAST'):
                self.map_graph_army.add_node(loc.upper())
            if game.map.loc_type[loc] in ('WATER', 'COAST'):
                self.map_graph_navy.add_node(loc.upper())

        locations = [loc.upper() for loc in locations]
        for i in locations:
            for j in locations:
                if game.map.abuts('A', i, '-', j):
                    self.map_graph_army.add_edge(i, j)
                if game.map.abuts('F', i, '-', j):
                    self.map_graph_navy.add_edge(i, j)

    # ------------------------------------------------------------------
    # Behavioural signal: update after each movement phase is processed
    # ------------------------------------------------------------------

    def update_after_movement(self, game, all_power_orders, self_locs_before,
                               self_units_before):
        """Call this ONLY after a Movement ('M') phase has just been
        processed (mirrors AttitudeAgent's own guard). game.get_order_status
        is only meaningful right after game.process() for that phase.

        self_locs_before / self_units_before must be captured BEFORE
        game.process() was called for this phase (same as AttitudeAgent
        does), since centres/units can change once orders resolve.
        """
        for power_name in self.behaviour_scores.keys():
            event_score = 0.0
            power_orders = all_power_orders.get(power_name, [])
            order_status = game.get_order_status(power_name=power_name)

            for order in power_orders:
                if '-' not in order and ' S ' not in order:
                    continue
                words = order.split(' ')
                unit = ' '.join(words[:2])
                success = order_status.get(unit, []) == []

                # Attack on one of our locations
                if '-' in order and len(words) in (4, 5):  # 5 = via convoy
                    target = words[words.index('-') + 1]
                    if target in self_locs_before:
                        event_score += 2.0 if success else 1.0

                # Support of one of our units -> lowers threat
                if ' S ' in order:
                    supported = ' '.join(
                        words[words.index('S') + 1: words.index('S') + 3])
                    if supported in self_units_before:
                        event_score -= 2.0 if success else 1.0

            event_score = max(-self.max_abs_event,
                               min(self.max_abs_event, event_score))

            old = self.behaviour_scores[power_name]
            self.behaviour_scores[power_name] = (
                self.decay * old + (1 - self.decay) * event_score
            )

    # ------------------------------------------------------------------
    # Positional signal: computed fresh each call, no history needed
    # ------------------------------------------------------------------

    def _border_pressure(self, game, self_locs, owner_of_loc):
        """For each opponent power, count how many of their units are
        within distance 1 of any of our locations. Cheap: only looks at
        immediate neighbours of our own locations, no full shortest-path
        search, so it stays comfortably inside the 1s budget even on a
        full board."""
        pressure = {p: 0 for p in self.behaviour_scores.keys()}

        neighbour_locs = set()
        for loc in self_locs:
            loc_u = loc.upper().split('/')[0]
            if loc_u in self.map_graph_army.nodes:
                neighbour_locs.update(self.map_graph_army.neighbors(loc_u))
            if loc_u in self.map_graph_navy.nodes:
                neighbour_locs.update(self.map_graph_navy.neighbors(loc_u))

        for loc in neighbour_locs:
            owner = owner_of_loc.get(loc)
            if owner and owner != self.power_name and owner in pressure:
                pressure[owner] += 1

        return pressure

    # ------------------------------------------------------------------
    # Combined, usable output
    # ------------------------------------------------------------------

    def get_normalised_threats(self, game):
        """Returns {power_name: threat in [0, 1]}, combining the decaying
        behavioural EMA with current border pressure, min-max normalised
        across the (up to 6) opponents so it's directly usable as a weight
        e.g. in a target-scoring function without needing to know the raw
        scale."""

        self_centers = game.get_centers(self.power_name)
        self_units_locs = game.get_orderable_locations(self.power_name)
        self_locs = list(set(self_centers + self_units_locs))

        owner_of_loc = {}
        for p in game.powers.keys():
            for loc in set(game.get_centers(p) + game.get_orderable_locations(p)):
                owner_of_loc[loc] = p

        pressure = self._border_pressure(game, self_locs, owner_of_loc)

        raw = {}
        for p, behav in self.behaviour_scores.items():
            raw[p] = behav + self.border_weight * pressure.get(p, 0)

        if not raw:
            return {}

        lo, hi = min(raw.values()), max(raw.values())
        if hi - lo < 1e-9:
            # No differentiation yet (e.g. turn 1) -> treat everyone as
            # equally, moderately unknown rather than dividing by zero.
            return {p: 0.5 for p in raw}

        return {p: (v - lo) / (hi - lo) for p, v in raw.items()}
