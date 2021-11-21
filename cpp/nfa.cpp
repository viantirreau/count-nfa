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

    for (auto i : _sorted_symbols)
    {
        cout << i << "\n";
    }
    cout << endl;
    compute_reverse_transitions();
    // remove_sink_states();
    // remove_unreachable_states();
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
        old_non_sink = non_sink; // Copy by value (I hope)
        for (auto p_a : _reverse_transitions)
        {
            auto p = p_a.first;
            if (!in(old_non_sink, p))
                continue;
            for (auto a_q : p_a.second)
            {
                for (int q : a_q.second)
                    non_sink.insert(q);
            }
        }
    }
    _states = intersect(_states, non_sink);
    _initial_states = intersect(_initial_states, non_sink);

    trans_t new_trans, new_rev_trans;
    for (auto p_a : _transitions)
    {
        auto p = p_a.first;
        if (!in(non_sink, p))
            continue;
        for (auto a_q : p_a.second)
        {
            auto a = a_q.first;
            for (int q : a_q.second)
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
        for (auto p_a : _transitions)
        {
            auto p = p_a.first;
            if (!in(old_reachable, p))
                continue;
            for (auto a_q : p_a.second)
            {
                for (int q : a_q.second)
                    all_reachable.insert(q);
            }
        }
    }
    _states = intersect(_states, all_reachable);
    _final_states = intersect(_final_states, all_reachable);
    trans_t new_trans, new_rev_trans;
    for (auto p_a : _transitions)
    {
        auto p = p_a.first;
        if (!in(all_reachable, p))
            continue;
        for (auto a_q : p_a.second)
        {
            auto a = a_q.first;
            for (int q : a_q.second)
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

uiset NFA::final_config(vector<int> input_str)
{
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
    return curr_states;
};

bool NFA::reachable(vector<int> input_str, int state)
{
    return in(NFA::final_config(input_str), state);
};