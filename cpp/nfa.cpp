#include <bits/stdc++.h>
#include "utils.cpp"
#include "nfa.hpp"
using namespace std;

#define uset unordered_set
#define uiset unordered_set<int>
#define iset set<int>
#define umap unordered_map
#define trans_t umap<int, map<int, iset>>

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
void NFA::remove_sink_states(){};
void NFA::remove_unreachable_states(){};
