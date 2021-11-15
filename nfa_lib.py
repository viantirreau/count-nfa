import abc
import copy
import math
from random import random, choices, shuffle, normalvariate
from collections import Counter, defaultdict
from turtle import distance

import numpy as np
import graphviz
from networkx import MultiDiGraph, DiGraph
from tqdm.auto import tqdm as tqdm
from networkx.algorithms.components import strongly_connected_components
from networkx.algorithms.shortest_paths.dense import floyd_warshall

"""
Classes and methods for working with nondeterministic finite automata.
Source: https://github.com/caleb531/automata
"""


def ddict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)


def retries_for_sample(kappa: int) -> int:
    return math.ceil(
        (2 + math.log(4) + 8 * math.log(kappa)) / math.log(1.0 / (1.0 - math.exp(-9.0)))
    )


class AutomatonException(Exception):
    """The base class for all automaton-related errors."""

    pass


class InvalidStateError(AutomatonException):
    """A state is not a valid state for this automaton."""

    pass


class InvalidSymbolError(AutomatonException):
    """A symbol is not a valid symbol for this automaton."""

    pass


class MissingStateError(AutomatonException):
    """A state is missing from the automaton definition."""

    pass


class RejectionException(AutomatonException):
    """The input was rejected by the automaton."""

    pass


class Automaton(metaclass=abc.ABCMeta):
    """An abstract base class for all automata, including Turing machines."""

    @abc.abstractmethod
    def __init__(self):
        """Initialize a complete automaton."""
        raise NotImplementedError

    @abc.abstractmethod
    def validate(self):
        """Return True if this automaton is internally consistent."""
        raise NotImplementedError

    @abc.abstractmethod
    def read_input_stepwise(self, input_str):
        """Return a generator that yields each step while reading input."""
        raise NotImplementedError

    def read_input(self, input_str, validate_final=True):
        """
        Check if the given string is accepted by this automaton.

        Return the automaton's final configuration. If validate_final is true,
        this will return the final configuration only if this string is valid,
        raising RejectionException otherwise.
        """
        validation_generator = self.read_input_stepwise(
            input_str, validate_final=validate_final
        )
        for config in validation_generator:
            pass
        return config

    def accepts_input(self, input_str):
        """Return True if this automaton accepts the given input."""
        try:
            self.read_input(input_str)
            return True
        except RejectionException:
            return False

    def _validate_initial_states(self):
        """Raise an error if any of the initial states is invalid."""
        invalid_states = self.initial_states - self.states
        if invalid_states:
            raise InvalidStateError(
                "inital states are not valid ({})".format(
                    ", ".join(str(state) for state in invalid_states)
                )
            )

    def _validate_initial_states_transitions(self):
        """Raise an error if any initial state has no transitions defined."""
        for initial_state in self.initial_states:
            if initial_state not in self.transitions:
                raise MissingStateError(
                    "initial state {} has no transitions defined".format(initial_state)
                )

    def _validate_final_states(self):
        """Raise an error if any final states are invalid."""
        invalid_states = self.final_states - self.states
        if invalid_states:
            raise InvalidStateError(
                "final states are not valid ({})".format(
                    ", ".join(str(state) for state in invalid_states)
                )
            )

    def copy(self):
        """Create a deep copy of the automaton."""
        return self.__class__(**vars(self))

    def __eq__(self, other):
        """Check if two automata are equal."""
        return vars(self) == vars(other)


class FA(Automaton, metaclass=abc.ABCMeta):
    """An abstract base class for finite automata."""

    pass


class NFA(FA):
    """A nondeterministic finite automaton."""

    def __init__(
        self,
        *,
        states,
        input_symbols,
        transitions,
        initial_states,
        final_states,
        states_by_layer=None,
        reverse_transitions=None,
    ):
        """Initialize a complete NFA."""
        self.states = states.copy()
        self.input_symbols = input_symbols.copy()
        self.transitions = copy.deepcopy(transitions)
        self.initial_states = initial_states
        self.final_states = final_states.copy()
        # self.validate()

        # --- Counting utilities ---
        self.states_by_layer = states_by_layer
        self.sketch = defaultdict(dict)
        self.n_for_sets = {}
        self.n_for_states = {}
        self.s_for_states = defaultdict(Counter)
        self.reverse_transitions = reverse_transitions
        self.sorted_symbols = sorted(self.input_symbols)
        self.remove_sink_states()
        self.remove_unreachable_states()

    def _validate_transition_invalid_symbols(self, start_state, paths):
        """Raise an error if transition symbols are invalid."""
        for input_symbol in paths.keys():
            if input_symbol not in self.input_symbols and input_symbol != "":
                raise InvalidSymbolError(
                    "state {} has invalid transition symbol {}".format(
                        start_state, input_symbol
                    )
                )

    def _validate_transition_end_states(self, start_state, paths):
        """Raise an error if transition end states are invalid."""
        for end_states in paths.values():
            for end_state in end_states:
                if end_state not in self.states:
                    raise InvalidStateError(
                        "end state {} for transition on {} is "
                        "not valid".format(end_state, start_state)
                    )

    def validate(self):
        """Return True if this NFA is internally consistent."""
        for start_state, paths in self.transitions.items():
            self._validate_transition_invalid_symbols(start_state, paths)
            self._validate_transition_end_states(start_state, paths)
        self._validate_initial_states()
        self._validate_initial_states_transitions()
        self._validate_final_states()
        return True

    def _get_next_current_states(self, current_states, input_symbol):
        """Return the next set of current states given the current set."""
        next_current_states = set()
        for current_state in current_states:
            symbol_end_states = self.transitions.get(current_state, {}).get(
                input_symbol
            )
            if symbol_end_states:
                for end_state in symbol_end_states:
                    next_current_states.add(end_state)

        return next_current_states

    def _check_for_input_rejection(self, current_states):
        """Raise an error if the given config indicates rejected input."""
        if not (current_states & self.final_states):
            raise RejectionException(
                "the NFA stopped on all non-final states ({})".format(
                    ", ".join(str(state) for state in current_states)
                )
            )

    def read_input_stepwise(self, input_str, validate_final=True):
        """
        Check if the given string is accepted by this NFA.

        Yield the current configuration of the NFA at each step.
        """
        # current_states = self._get_epsilon_closure(self.initial_state)
        current_states = self.initial_states

        yield current_states
        for input_symbol in input_str:
            current_states = self._get_next_current_states(current_states, input_symbol)
            yield current_states

        if validate_final:
            self._check_for_input_rejection(current_states)

    def reachable(self, input_str: str, state) -> bool:
        """
        Returns true if the NFA reaches the state after reading input_str
        """
        return bool(self.read_input(input_str, validate_final=False) & set([state]))

    @staticmethod
    def remove_unreachable_states_n_aware(
        initial_states: set, transitions: dict, n: int
    ):
        """
        Filters states that are unreachable from an initial state
        by traversing one layer at a time.

        It returns a tuple of the form
        (
            all_reachable_states: set of states,
            reachable_states_by_layer: dict of layer -> set of states,
            reachable_transitions: dict of state p -> symbol a -> state q,
            rev_reachable_transitions: dict of state q -> symbol a -> state p,
        )

        Note that here states are tuples of strings (former states) and layer integers.
        """
        reachable_states_by_layer = {0: initial_states}
        all_reachable_states = initial_states.copy()
        for next_layer in range(1, n + 1):
            this_layer_reachable = reachable_states_by_layer[next_layer - 1]
            next_layer_reachable = set()
            for current_state in this_layer_reachable:
                for new_states in transitions[current_state].values():
                    for new_state in new_states:
                        next_layer_reachable.add(new_state)
            all_reachable_states.update(next_layer_reachable)
            reachable_states_by_layer[next_layer] = next_layer_reachable

        reachable_transitions = defaultdict(lambda: defaultdict(set))
        rev_reachable_transitions = defaultdict(lambda: defaultdict(set))

        for p, trans in transitions.items():
            if p not in all_reachable_states:
                # print(f"Unreachable state {p} was removed")
                continue
            for a, qs in trans.items():
                for q in qs:
                    if q not in all_reachable_states:
                        # print(f"Unreachable state {q} was removed")
                        continue
                    reachable_transitions[p][a].add(q)
                    rev_reachable_transitions[q][a].add(p)

        return (
            all_reachable_states,
            reachable_states_by_layer,
            reachable_transitions,
            rev_reachable_transitions,
        )

    def compute_reverse_transitions(self):
        # Compute the reverse transitions:
        rev = defaultdict(lambda: defaultdict(set))
        for p, trans in self.transitions.items():
            for a, qs in trans.items():
                for q in qs:
                    rev[q][a].add(p)
        self.reverse_transitions = rev

    def remove_unreachable_states(self):
        """
        Removes the states that are never reached
        from an initial state.
        In-place operation that modifies the states,
        final_states and transitions.
        """
        all_reachable = self.initial_states.copy()
        old_reachable = set()
        while old_reachable != all_reachable:
            old_reachable = all_reachable.copy()
            for p, trans in self.transitions.items():
                if p not in old_reachable:
                    continue
                for a, qs in trans.items():
                    for q in qs:
                        all_reachable.add(q)
        self.states = self.states & all_reachable
        self.final_states = self.final_states & all_reachable
        new_transitions = defaultdict(lambda: defaultdict(set))
        new_rev_transitions = defaultdict(lambda: defaultdict(set))
        for p, trans in self.transitions.items():
            if p not in all_reachable:
                continue
            for a, qs in trans.items():
                for q in qs:
                    if q not in all_reachable:
                        continue
                    new_transitions[p][a].add(q)
                    new_rev_transitions[q][a].add(p)
        self.transitions = ddict2dict(new_transitions)
        self.reverse_transitions = ddict2dict(new_rev_transitions)

    def remove_sink_states(self):
        """
        Removes the states that never reach a final state.
        In-place operation that modifies the states,
        initial_states and transitions.
        """
        if self.reverse_transitions is None:
            self.compute_reverse_transitions()

        non_sink = self.final_states.copy()
        old_non_sink = set()
        while old_non_sink != non_sink:
            old_non_sink = non_sink.copy()
            for p, trans in self.reverse_transitions.items():
                if p not in old_non_sink:
                    continue
                for a, qs in trans.items():
                    for q in qs:
                        non_sink.add(q)
        self.states = self.states & non_sink
        self.initial_states = self.initial_states & non_sink
        new_transitions = defaultdict(lambda: defaultdict(set))
        new_rev_transitions = defaultdict(lambda: defaultdict(set))
        for p, trans in self.transitions.items():
            if p not in non_sink:
                continue
            for a, qs in trans.items():
                for q in qs:
                    if q not in non_sink:
                        continue
                    new_transitions[p][a].add(q)
                    new_rev_transitions[q][a].add(p)
        self.transitions = ddict2dict(new_transitions)
        self.reverse_transitions = ddict2dict(new_rev_transitions)

    def unroll(self, n: int):
        """
        Builds A_unroll with n levels
        to estimate |L(F^n)|
        """
        # for each state q ∈ Q, include copies q_0 , q_1 , ..., q n in A unroll
        new_states = {(q, i) for q in self.states for i in range(1, n + 1)}
        # new_states_by_layer = {i: {(q,i) for q in self.states} for i in range(1, n+1)}
        new_states.update({(q, 0) for q in self.initial_states})
        # new_states_by_layer = {0: {(q,0) for q in self.initial_states}}

        # for each transition (p, a, q) ∈ ∆ and i ∈ {0, 1, . . . , n − 1}, include
        # transition (p_i, a, q_i+1) in A unroll
        new_transitions = defaultdict(lambda: defaultdict(set))
        rev_transitions = defaultdict(lambda: defaultdict(set))
        for p, trans in self.transitions.items():
            for a, qs in trans.items():
                for q in qs:
                    for i in range(n):
                        # state -> symbol -> set of states
                        new_transitions[p, i][a].add((q, i + 1))
                        rev_transitions[q, i + 1][a].add((p, i))

        new_initial_states = {(q, 0) for q in self.initial_states}
        new_final_states = {(q, n) for q in self.final_states}

        new_transitions = ddict2dict(new_transitions)
        rev_transitions = ddict2dict(rev_transitions)
        nfa_unroll = NFA(
            states=new_states,
            input_symbols=self.input_symbols,
            transitions=new_transitions,
            initial_states=new_initial_states,
            final_states=new_final_states,
            reverse_transitions=rev_transitions,
        )
        # Update the states_by_layer
        states_by_layer = defaultdict(set)
        for state, layer in nfa_unroll.states:
            states_by_layer[layer].add((state, layer))
        nfa_unroll.states_by_layer = ddict2dict(states_by_layer)
        return nfa_unroll

    def compute_n_for_single_state(self, state):
        """
        Compute N(pᵅ) = Σ N(R_b) for R_b being the set of
        incoming states which connect to pᵅ after reading symbol b.

        In simple terms, it receives a states and estimates
        the number of different strings leading to it.
        """

        if state in self.n_for_states:
            return self.n_for_states[state]

        # will store all the N(Pᵅ) calculated with `compute_n_for_states_set`
        n_r_b = {}
        for b, leading_states in self.reverse_transitions.get(state, {}).items():
            # leading_states = self.reverse_transitions[state][b]
            n_r_b[b] = self.compute_n_for_states_set(frozenset(leading_states))
        n_q_alpha = sum(n_r_b.values())
        self.n_for_states[state] = n_q_alpha
        return n_q_alpha

    def compute_n_for_states_set(self, states: frozenset):
        """
        Compute N(Pᵅ) = Σ N(pᵅ)· |S(pᵅ)-Union(L(qᵅ) for q ≺ p)| / |S(pᵅ)|

        In simple terms, it receives a set of states and estimates
        the number of different strings leading to all of them.
        """
        if not states:
            return 0
        if states in self.n_for_sets:
            return self.n_for_sets[states]

        states_list = sorted(states)  # linear order ≺
        # Count the first state in the list alone
        total = self.n_for_states[states_list[0]]
        for i in range(1, len(states_list)):
            # states[i] = (q, i), where q is the original state name in A
            anchor_state = states_list[i]
            intersection_count = 0
            s_size = 0
            # Now estimate the intersection rate for anchor_state
            for string, count in self.s_for_states[anchor_state].items():
                s_size += count
                was_reachable = False
                for previous_state in states_list[:i]:
                    if self.reachable(input_str=string, state=previous_state):
                        was_reachable = True
                        break
                # whether string is not in L(q_i) for every q_i < anchor
                if not was_reachable:
                    intersection_count += count
            intersection_rate = intersection_count / s_size if s_size > 0 else 0
            total += self.compute_n_for_single_state(anchor_state) * intersection_rate
        # cache the result
        self.n_for_sets[states] = total
        return total

    def sample(self, beta: int, states: frozenset, curr_string: str, phi: float):
        if beta == 0:
            if random() <= phi:
                return curr_string
            return "fail"
            # return curr_string
        # will store all the states on layer i-1
        # which lead to r after reading b={0,1}
        # {'0': {...leading_states}, '1': {...leading_states}}
        p_beta_b = {}
        # will store all the N(Pᵅ) calculated with `compute_n_for_states_set`
        n_p_beta_b = {}

        for b in self.sorted_symbols:
            all_leading_states = set()
            for r in states:
                all_leading_states.update(
                    self.reverse_transitions.get(r, {}).get(b, set())
                )
            fzset = frozenset(all_leading_states)
            p_beta_b[b] = fzset
            n_p_beta_b[b] = self.compute_n_for_states_set(fzset)
        sum_n_p_beta = sum(n_p_beta_b.values())
        if sum_n_p_beta == 0:
            return None
        chosen_symbol = choices(
            self.sorted_symbols,
            weights=[n_p_beta_b[b] / sum_n_p_beta for b in self.sorted_symbols],
            k=1,
        )[0]
        curr_string = chosen_symbol + curr_string  # w_beta-1 = b · w_beta
        chosen_states = p_beta_b[chosen_symbol]  # p_beta-1
        new_probability = phi / (n_p_beta_b[chosen_symbol] / sum_n_p_beta)  # phi / p_b
        return self.sample(
            beta=beta - 1,
            states=chosen_states,
            curr_string=curr_string,
            phi=new_probability,
        )

    def count_accepted(self, n: int, eps: float = 1.0, kappa_multiple: int = 1):
        """
        Returns a (1 ± ε)-approximation of |L_n(A_unroll)|
        """
        kappa = math.ceil(n * len(self.states) / eps)
        # c(κ)
        retries_sample = retries_for_sample(kappa)
        print(f"Retries per sample {retries_sample}")
        sample_size = kappa_multiple * kappa  # 2 * kappa ** 7
        print(f"Sample size {sample_size}")
        exp_minus_five = math.exp(-5)
        # For each state q ∈ I, set N(q_0) = |L(q_0)| = 1
        # and S(q_0) = L(q_0) = {λ}
        for q in self.states_by_layer[0]:
            self.s_for_states[q] = {"": sample_size}
            self.n_for_states[q] = 1

        # For each i = 1, . . . , n and state q ∈ Q:
        #   (a) Compute N(q i ) given sketch[i − 1]
        #   (b) Sample polynomially many uniform elements from L(q_i) using
        #       N(q_i) and sketch[i − 1], and let S(q_i) be the multiset of
        #       uniform samples obtained
        for i in tqdm(range(1, n + 1), desc="Layer", position=0):
            for q in tqdm(
                self.states_by_layer[i],
                desc=f"State at layer {i}",
                position=1,
                leave=False,
            ):

                n_q_alpha = self.compute_n_for_single_state(q)
                if n_q_alpha == 0:
                    raise ValueError(f"Got 0 when computing n_q_alpha for {q=}")
                this_q_samples = Counter()
                # sample probability
                phi = exp_minus_five / n_q_alpha
                for _ in tqdm(
                    range(sample_size), desc="Sampling", position=2, leave=False
                ):
                    sampled_successfully = False
                    for retry in range(retries_sample):
                        potential_sample = self.sample(
                            beta=i, states=frozenset([q]), curr_string="", phi=phi
                        )
                        if potential_sample == "fail":
                            continue
                        elif potential_sample is None:
                            continue
                        this_q_samples.update([potential_sample])
                        sampled_successfully = True
                        break
                    if not sampled_successfully:
                        return 0

                self.s_for_states[q] = this_q_samples

        return self.compute_n_for_states_set(frozenset(self.final_states))

    def bruteforce_save_all(self, n: int):
        if n == 0:
            return [""] if self.accepts_input("") else []
        accepted_strings = []
        for i in range(2 ** n):
            string_i = bin(i)[2:].zfill(n)
            if self.accepts_input(string_i):
                accepted_strings.append(string_i)
        return accepted_strings

    def bruteforce_count_only(self, n: int):
        if n == 0:
            return 1 if self.accepts_input("") else 0
        accepted_count = 0
        for i in tqdm(range(2 ** n), leave=False):
            string_i = bin(i)[2:].zfill(n)
            if self.accepts_input(string_i):
                accepted_count += 1
        return accepted_count

    def bruteforce_dfs(self, n: int):
        if n == 0:
            return 1 if self.accepts_input("") else 0
        accepted_count = 0
        visit_queue = [(s, 0) for s in self.initial_states]  # FIFO queue
        while visit_queue:
            curr_state, curr_len = visit_queue.pop(-1)
            if curr_len == n:
                if curr_state in self.final_states:
                    accepted_count += 1
                continue
            for symbol, next_states in self.transitions.get(curr_state, {}).items():
                for next_state in next_states:
                    visit_queue.append((next_state, curr_len + 1))
        return accepted_count

    def plot(self):
        dot = graphviz.Digraph(
            name="NFA", graph_attr={"rankdir": "LR", "ranksep": "0.5", "nodesep": "0.6"}
        )

        for s in self.states:
            if s in self.final_states:
                dot.node(str(s), shape="doublecircle")
            else:
                dot.node(str(s), shape="circle")

            if s in self.initial_states:
                dot.node(f"{s}_init", label="", shape="none", height="0", width="0")
                dot.edge(f"{s}_init", str(s))

        for curr_state, symbol_next_states in self.transitions.items():
            defer_labels = []
            for symbol, next_states in symbol_next_states.items():
                for next_state in next_states:
                    if next_state == curr_state:
                        defer_labels.append(symbol)
                        continue
                    dot.edge(str(curr_state), str(next_state), label=symbol)
            if defer_labels:
                dot.edge(str(curr_state), str(curr_state), label=",".join(defer_labels))

        return dot

    @staticmethod
    def random_trans_sparse(
        states_list: list, input_symbols: set = {"0", "1"}, sparsity=0.5
    ):
        """
        Generates random transitions for the states
        q_0 ... q_{n_states}, using the input_symbols.

        Each pair of states can be connected with up to
        one symbol transition, as the symbol is
        chosen at random to form a transition matrix.
        """
        n_states = len(states_list)
        # choose the transitions as a random matrix over the alphabet and the
        # empty symbol, represented by -1
        matrix_choices = list(input_symbols | {"-1"})
        transition_matrix = np.random.choice(matrix_choices, size=(n_states, n_states))
        transitions = {}
        for i, from_state in enumerate(states_list):
            transitions_from_state = defaultdict(set)
            for j, to_state in enumerate(states_list):
                if transition_matrix[i, j] in input_symbols:
                    symbol = str(transition_matrix[i, j])
                    transitions_from_state[symbol].add(to_state)
            transitions[from_state] = transitions_from_state

        return transitions

    @staticmethod
    def random_trans_ambiguous(
        states_list: list, input_symbols: set = {"0", "1"}, sparsity: float = 1 / 3
    ):
        """
        Generates random transitions for the states
        q_0 ... q_{n_states}, using the input_symbols.

        Each pair of states may be connected with every
        available symbol, as a subset is chosen at
        random for each pair.
        """

        transitions = {}
        for from_state in states_list:
            transitions_from_state = defaultdict(set)
            for to_state in states_list:
                if random() < sparsity:
                    # no transition case
                    continue
                # each symbol has constant probability 0.5 of being part of a
                # transition from_state->to_state
                for symbol in input_symbols:
                    if random() < 0.5:
                        transitions_from_state[symbol].add(to_state)

                """
                # Alternative approach
                
                # Note that by choosing the subset size uniformly at random,
                # we are flattening the distribution of a random subset in
                # which each element is included in the subset with p=0.5.
                # That approach will tend to a binomial distribution on the
                # subset size, which in turn tends to a normal distribution
                # for large alphabets
                subset_size = randint(1, len(input_symbols))
                shuffle(input_list)
                subset = input_list[:subset_size]
                for symbol in subset:
                    transitions_from_state[symbol].add(to_state)
                        
                """
            transitions[from_state] = transitions_from_state
        return transitions

    @staticmethod
    def random(n_states: int, input_symbols: set = {"0", "1"}, sparsity: float = 1 / 3):
        """
        Calls NFA.random_trans_? to generate random transitions,
        and generates a random subset of states for
        the initial and final states.

        Returns the resulting random NFA instance.
        """

        states_list = [f"q{i}" for i in range(n_states)]
        states = set(states_list)

        transitions = NFA.random_trans_sparse(
            states_list=states_list, input_symbols=input_symbols, sparsity=sparsity
        )

        shuffle(states_list)
        n_initial_states = max(
            1, min(n_states, int(normalvariate(1, (n_states ** 0.5) / 2)))
        )
        initial_states = set(states_list[:n_initial_states])
        shuffle(states_list)
        n_final_states = max(
            1, min(n_states, int(normalvariate(1, (n_states ** 0.5) / 2)))
        )
        final_states = set(states_list[:n_final_states])

        return NFA(
            states=states,
            input_symbols=input_symbols,
            transitions=transitions,
            initial_states=initial_states,
            final_states=final_states,
        )

    def to_networkx(self):
        """
        Returns the NFA transitions as a NetworkX DiGraph.
        """
        graph = MultiDiGraph()
        graph.add_nodes_from(self.states)
        for from_state, symbol_to_states in self.transitions.items():
            for symbol, to_states in symbol_to_states.items():
                for to_state in to_states:
                    graph.add_edge(from_state, to_state, symbol=symbol)
        return graph

    def cycle_height(self):
        """
        Computes the cycle height using the algorithm proposed by C. Keeler and K. Salomaa in
        https://doi.org/10.1016/j.ic.2021.104690
        """
        nfa_nx = self.to_networkx()
        sccs = list(strongly_connected_components(nfa_nx))
        # Check that all sccs are either an acyclic singleton
        # (consisting of only one state and no transitions) or
        # a unique simple cycle
        state_scc_idx_map = {}
        acyclic_singleton_idx_map = {}
        for scc_idx, scc in enumerate(sccs):
            for state in scc:
                state_scc_idx_map[state] = scc_idx
                outgoing_count = 0
                for outgoing in self.transitions.get(state, {}).values():
                    # add one to the counter every time we see that
                    # an outgoing edge is in the scc
                    outgoing_count += len(outgoing & scc)
                # If any state has two different paths for states in
                # the same scc, then a nested cycle is diverging from
                # this state. Note that a self loop for two or more
                # symbols will also count as more than one `outgoing`
                if outgoing_count > 1:
                    return float("inf")
                acyclic_singleton_idx_map[scc_idx] = outgoing_count == 0

        distance_matrix = floyd_warshall(nfa_nx)
        scc_graph = DiGraph()
        scc_graph.add_nodes_from(range(len(sccs)))

        for scc_idx_i in range(len(sccs)):
            for scc_idx_j in range(len(sccs)):
                if scc_idx_i == scc_idx_j:
                    continue
                scc_i = sccs[scc_idx_i]
                scc_j = sccs[scc_idx_j]
                for state_i in scc_i:
                    for state_j in scc_j:
                        if distance_matrix[state_i][state_j] < float("inf"):
                            # Is an acyclic singleton
                            if len(scc_j) == 1:
                                # Add a 0-weight edge to E from vertex scc_idx_i
                                # to vertex scc_idx_j
                                scc_graph.add_edge(scc_idx_i, scc_idx_j, weight=0)
                            else:
                                # It's a unique simple cycle, add a 1-weight edge
                                scc_graph.add_edge(scc_idx_i, scc_idx_j, weight=1)
        scc_distance_matrix: dict = floyd_warshall(scc_graph)
        max_cycle_height = 0
        for initial_state in self.initial_states:
            scc_idx_for_init_state = state_scc_idx_map[initial_state]
            start_bias = 1
            # if this start state is in an acyclic singleton
            if acyclic_singleton_idx_map[scc_idx_for_init_state]:
                # start bias should be 0
                start_bias = 0
            max_dist_for_init_state = max(
                filter(
                    lambda x: x < float("inf"),
                    scc_distance_matrix[scc_idx_for_init_state].values(),
                )
            )
            max_cycle_height = max(
                max_cycle_height, start_bias + max_dist_for_init_state
            )
        return max_cycle_height

    @staticmethod
    def from_random_matrix(matrix: np.ndarray) -> "NFA":
        n_states = matrix.shape[1]
        alph_size = (matrix.shape[0] - 2) // n_states  # normally 2
        input_symbols_list = [str(i) for i in range(alph_size)]
        input_symbols = set(input_symbols_list)
        states_list = [str(i) for i in range(n_states)]
        transitions = defaultdict(dict)
        for idx, symbol in enumerate(input_symbols_list):
            square_matrix = matrix[idx * n_states : (idx + 1) * n_states, :]
            for i, from_state in enumerate(states_list):
                transitions_from_state = defaultdict(set)
                for j, to_state in enumerate(states_list):
                    if square_matrix[i, j] == 1:
                        transitions_from_state[symbol].add(to_state)
                transitions[from_state].update(transitions_from_state)

        initial_states = set()
        for idx in range(n_states):
            if matrix[-2, idx] == 1:
                initial_states.add(states_list[idx])
        final_states = set()
        for idx in range(n_states):
            if matrix[-1, idx] == 1:
                final_states.add(states_list[idx])
        return NFA(
            states=set(states_list),
            input_symbols=input_symbols,
            initial_states=initial_states,
            final_states=final_states,
            transitions=transitions,
        )


def count_nfa(nfa: NFA, n: int, eps: float = 1, kappa_multiple: int = 1):
    """
    Unrolls the given NFA and returns a tuple with the estimated count of
    accepted strings of length n, as well as the unrolled NFA.

    Returns: Tuple(count_accepted: int, nfa_unroll: NFA)
    """
    nfa_unroll = nfa.unroll(n)
    return (
        nfa_unroll.count_accepted(n=n, eps=eps, kappa_multiple=kappa_multiple),
        nfa_unroll,
    )


def random_matrix_for_nfa(
    n_states: int, sparsity: float, n_initial: int, n_final: int, alph_size: int = 2
):
    """
    Generates a random binary matrix of the form

    ---------------------------------------------------

      (n_states, n_states) -> transitions for symbol 0
         A 1-cell represents a transition from
           state i to state j by reading a 0
        (sparsity = number of 0s / number of 1s)

    ---------------------------------------------------

      (n_states, n_states) -> transitions for symbol 1

    ---------------------------------------------------
      (n_states, ) -> random vector with n_initial 1s
    ---------------------------------------------------
      (n_states, ) -> random vector with n_final 1s
    ---------------------------------------------------
    """
    assert 0 <= sparsity <= 1
    assert n_initial <= n_states
    assert n_final <= n_states
    output = []
    for _ in range(alph_size):
        output.append(
            np.random.choice([0, 1], (n_states, n_states), p=[sparsity, 1 - sparsity])
        )
    initial_row = np.array([1] * n_initial + [0] * (n_states - n_initial))
    final_row = np.array([1] * n_final + [0] * (n_states - n_final))
    np.random.shuffle(initial_row)
    np.random.shuffle(final_row)
    output += [initial_row.reshape(1, -1), final_row.reshape(1, -1)]
    return np.concatenate(output)
