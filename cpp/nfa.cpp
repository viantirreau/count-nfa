#include <bits/stdc++.h>
#include "utils.cpp"
#include "nfa.hpp"
using namespace std;

#define uset unordered_set
#define uiset unordered_set<int>
#define iset set<int>
#define umap unordered_map
#define trans_t umap<int, map<int, uiset>>

NFA::NFA(uiset states, uiset input_symbols,
         trans_t transitions,
         uiset initial_states,
         uiset final_states) : _states(states),
                               _input_symbols(input_symbols),
                               _transitions(transitions),
                               _initial_states(initial_states),
                               _final_states(final_states)
{
    // convert uiset to vector, save in this->_sorted_symbols
    copy(input_symbols.begin(), input_symbols.end(), back_inserter(_sorted_symbols));
    sort(_sorted_symbols.begin(), _sorted_symbols.end());
    compute_reverse_transitions();
    remove_sink_states();
    remove_unreachable_states();
}

void NFA::compute_reverse_transitions()
{
    trans_t rev;
    for (auto curr_state : _transitions)
    {
        for (auto symbol_next_state : curr_state.second)
        {
            for (int next_state : symbol_next_state.second)
            {
                rev[next_state][symbol_next_state.first].insert(curr_state.first);
            }
        }
    }
    _reverse_transitions = rev;
}
void NFA::remove_sink_states()
{
    uiset old_non_sink, non_sink = _final_states;
    while (old_non_sink != non_sink)
    {
        old_non_sink = non_sink; // Copy by value
        // Iterate over (p,a,q) tuples on the reverse transitions
        for (auto &[p, a_qs] : _reverse_transitions)
        {
            if (!in(old_non_sink, p))
                continue;
            for (auto &[a, qs] : a_qs)
            {
                for (int q : qs)
                    non_sink.insert(q);
            }
        }
    }
    _states = intersect(_states, non_sink);
    _initial_states = intersect(_initial_states, non_sink);

    trans_t new_trans, new_rev_trans;
    // Rebuild the transitions using only the non_sink states
    for (auto &[p, a_qs] : _transitions)
    {
        if (!in(non_sink, p))
            continue;
        for (auto &[a, qs] : a_qs)
        {
            for (int q : qs)
            {
                if (!in(non_sink, q))
                    continue;
                new_trans[p][a].insert(q);
                new_rev_trans[q][a].insert(p);
            }
        }
    }
    _transitions = new_trans;
    _reverse_transitions = new_rev_trans;
};

void NFA::remove_unreachable_states()
{
    uiset old_reachable, all_reachable = _initial_states;
    while (old_reachable != all_reachable)
    {
        old_reachable = all_reachable;
        // Iterate over (p,a,q) tuples on the transitions
        for (auto &[p, a_qs] : _transitions)
        {
            if (!in(old_reachable, p))
                continue;
            for (auto &[a, qs] : a_qs)
            {
                for (int q : qs)
                    all_reachable.insert(q);
            }
        }
    }
    _states = intersect(_states, all_reachable);
    _final_states = intersect(_final_states, all_reachable);
    trans_t new_trans, new_rev_trans;
    // Rebuild the transitions using only the reachable states
    for (auto &[p, a_qs] : _transitions)
    {
        if (!in(all_reachable, p))
            continue;
        for (auto &[a, qs] : a_qs)
        {
            for (int q : qs)
            {
                if (!in(all_reachable, q))
                    continue;
                new_trans[p][a].insert(q);
                new_rev_trans[q][a].insert(p);
            }
        }
    }
    _transitions = new_trans;
    _reverse_transitions = new_rev_trans;
};

uiset NFA::final_config(const vector<int> &input_str)
{
    // Read the string one symbol at a time
    // and keep track of the current states.
    uiset curr_states = _initial_states;
    for (int symbol : input_str)
    {
        uiset next_curr_states;
        for (auto state : curr_states)
        {
            uiset reached_states;
            if (in(_transitions, state) && in(_transitions[state], symbol))
                reached_states = _transitions[state][symbol];
            if (!reached_states.empty())
                next_curr_states.insert(reached_states.begin(), reached_states.end());
        }
        curr_states = next_curr_states;
    }
    // Final configuration after reading the whole string
    return curr_states;
};

bool NFA::reachable(const vector<int> &input_str, int state)
{
    return in(NFA::final_config(input_str), state);
};

bool NFA::accepts(const vector<int> &input_str)
{
    return not_empty_intersection(NFA::final_config(input_str), _final_states);
}

vector<int> int_to_binary(int number, int length)
{
    // Initialize result with `length` zeros
    vector<int> result(length);
    for (int bit_idx = 0; bit_idx < length; bit_idx++)
        result[length - bit_idx - 1] = (number >> bit_idx) & 1;
    return result;
}

long long NFA::bruteforce_count_only(int n)
{
    long long count = 0;
    long long max_string_id = 1 << n;
    long long string_id;
    vector<int> string;
    for (string_id = 0; string_id < max_string_id; string_id++)
    {
        string = int_to_binary(string_id, n);
        if (accepts(string))
            count++;
    }
    return count;
}

NFA NFA::unroll(int n)
{
    uiset new_states, new_init_states, new_final_states;
    umap<int, iset> states_by_layer;

    // Map of the form {old_q, layer} => new_q
    map<pair<int, int>, int> old_state_layer_to_new_states;
    map<int, pair<int, int>> new_state_to_old_state_layer;
    int new_state_id = 0;
    // Fill the first layer with _initial_states
    // and compute the new state ids
    for (auto init_state : _initial_states)
    {
        // Create the state
        new_states.insert(new_state_id);
        // Compute the new initial states
        new_init_states.insert(new_state_id);
        // Update the states by layer
        states_by_layer[0].insert(new_state_id);
        // And update the reverse map
        new_state_to_old_state_layer[new_state_id] = {init_state, 0};
        old_state_layer_to_new_states[{init_state, 0}] = new_state_id++;
    }
    // Fill the following layers. No transitions have been computed yet
    for (int layer = 1; layer <= n; layer++)
    {
        for (auto state : _states)
        {
            // Create the state
            new_states.insert(new_state_id);
            // Update the states by layer
            states_by_layer[layer].insert(new_state_id);
            // Compute the new final states if needed
            if (layer == n)
                new_final_states.insert(new_state_id);
            // And update the reverse map
            new_state_to_old_state_layer[new_state_id] = {state, layer};
            old_state_layer_to_new_states[{state, layer}] = new_state_id++;
        }
    }
    // Compute just the new transitions (the reverse ones are calculated on
    // the constructor NFA::NFA)
    trans_t new_trans;
    // Iterate over (p,a,q) tuples on the transitions
    for (auto &[p, a_qs] : _transitions)
    {
        for (auto &[a, qs] : a_qs)
        {
            for (int q : qs)
            {
                for (int i = 0; i <= n; i++)
                {
                    // Insert the [(p, i)][a] -> (q, i+1) into the new transitions
                    int new_p_layer_i = old_state_layer_to_new_states[{p, i}];
                    int new_q_layer_i_1 = old_state_layer_to_new_states[{q, i + 1}];
                    new_trans[new_p_layer_i][a].insert(new_q_layer_i_1);
                }
            }
        }
    }
    NFA new_nfa = NFA(new_states, _input_symbols, new_trans, new_init_states, new_final_states);
    new_nfa._states_by_layer = states_by_layer;
    new_nfa._pre_unroll_state_map = new_state_to_old_state_layer;
    return new_nfa;
}