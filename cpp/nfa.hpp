#pragma once
#include <bits/stdc++.h>
using namespace std;

#define uset unordered_set
#define uiset unordered_set<int>
#define iset set<int>
#define umap unordered_map
#define trans_t umap<int, map<int, uiset>>
#define ll long long
class NFA
{
public:
    // States will be modeled as integers
    // Input symbols will be integers as well
    uiset _states, _input_symbols;
    // Transitions map from an int state
    // to another map from an int symbol,
    // finally to the new int state
    trans_t _transitions, _reverse_transitions;
    // Initial and final states
    uiset _initial_states, _final_states;
    // Storing the states by layer makes it
    // easy to iterate over the layers
    umap<int, iset> _states_by_layer;
    // Store the sampled predictions
    // for one state
    umap<int, double> _n_for_states;
    // Store the sampled predictions
    // for a set of states
    // This **has to be** a map,
    // can't be an unordered_map
    // because sets are not hashable
    map<iset, double> _n_for_sets;
    // Store the counts for every
    // sampled string, mapping from
    // a state to strings (vector of ints)
    // and their respective sampled counts
    umap<int, map<vector<int>, ll>> _s_for_states;
    // Sorted symbols for iteration and sampling
    vector<int> _sorted_symbols;
    // Map of the form new_q => {old_q, layer}
    // Useful for debugging the unroll operation
    map<int, pair<int, int>> _pre_unroll_state_map;
    // Constructor for NFA(Q, Σ, Δ, I, F)
    NFA(uiset states, uiset input_symbols,
        trans_t transitions,
        uiset initial_states,
        uiset final_states);
    // Handle dirty NFAs with useless states
    /** 
     * Removes the states that never reach a final state. 
     * In-place operation that modifies the states, 
     * initial_states and transitions.
     **/
    void remove_sink_states();
    /** 
     * Removes the states that are never reached
     * from an initial state.
     * In-place operation that modifies the states,
     * final_states and transitions.
     **/
    void remove_unreachable_states();
    // Fill the _reverse_transitions, useful in later computations
    void compute_reverse_transitions();
    // Reachability
    /**
     * Return the automaton's final configuration after reading 
     * the given string.
     */
    uiset final_config(const vector<int> &input_str);
    /**
     * Returns true if the NFA reaches the state after reading input_str.
     * 
     * @param input_str 
     * @param state 
     * @return true | false
     */
    bool reachable(const vector<int> &input_str, int state);

    /**
     * @brief Returns true if the automaton
     * accepts the given string.
     * 
     * @param input_str 
     * @return true 
     * @return false 
     */
    bool accepts(const vector<int> &input_str);

    // Randomized counting
    /**
     * @brief Builds A_unroll with n layers 
     * to estimate |L(F^n)|
     * @param n 
     * @return NFA 
     */
    NFA unroll(int n);
    /**
     * @brief 
     * Compute N(pᵅ) = Σ N(R_b) for R_b being the set of 
     * incoming states which connect to pᵅ after reading symbol b.
     *  
     * In simple terms, it receives a states and estimates
     * the number of different strings leading to it.
     * @param state 
     * @return long long 
     */
    double compute_n_for_single_state(int state);
    /**
     * @brief 
     * Compute N(Pᵅ) = Σ N(pᵅ)· |S(pᵅ)-Union(L(qᵅ) for q ≺ p)| / |S(pᵅ)| 
     * In simple terms, it receives a set of states and estimates 
     * the number of different strings leading to all of them.
     * @param states 
     * @return long long
     */
    double compute_n_for_states_set(uiset states);

    vector<int> sample(int beta, uiset states, vector<int> curr_string, float phi);
    // Returns a (1 ± ε)-approximation of |L_n(A_unroll)|
    double count_accepted(int n, float epsilon, int kappa_multiple);
    // Deterministic counting
    ll bruteforce_count_only(int n);
};