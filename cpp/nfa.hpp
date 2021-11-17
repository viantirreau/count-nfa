#include <bits/stdc++.h>
using namespace std;

#define uset unordered_set
#define umap unordered_map
class NFA
{
private:
    /* data */
public:
    // States will be modeled as integers
    uset<int> states;
    // Input symbols will be integers as well
    uset<int> input_symbols;
    uset<int> initial_states;
    uset<int> final_states;
    // Transitions map from an int state
    // to another map from an int symbol,
    // finally to the new int state
    umap<int, map<int, set<int>>> transitions;
    umap<int, map<int, set<int>>> reverse_transitions;
    // Storing the states by layer makes it
    // easy to iterate over the layers
    umap<int, set<int>> states_by_layer;
    // Store the sampled predictions
    // for one state
    umap<int, int> n_for_states;
    // Store the sampled predictions
    // for a set of states
    umap<set<int>, int> n_for_sets;
    // Store the counts for every
    // sampled string, mapping from
    // a state to strings (vector of ints)
    // and their respective sampled counts
    umap<int, map<vector<int>, int>> s_for_states;

    NFA(/* args */);
    ~NFA();

    void remove_sink_states();
    void remove_unreachable_states();
    bool reachable(vector<int> input_str, int state);
};
