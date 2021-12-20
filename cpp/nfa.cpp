#include <bits/stdc++.h>
#include "utils.cpp"
#include "nfa.hpp"

using namespace std;

#define uset unordered_set
#define uiset unordered_set<int>
#define iset set<int>
#define umap unordered_map
#define trans_t umap<int, map<int, uiset>>
#define ll long long

// Will be used to obtain a seed for the random number engine
random_device rd;
// Standard mersenne_twister_engine seeded with rd()
mt19937_64 gen(rd());
uniform_real_distribution rand_zero_one(0.0, 1.0);
ll sampled_symbols = 0;

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

uiset NFA::final_config(const deque<int> &input_str)
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

bool NFA::reachable(const deque<int> &input_str, int state)
{
    uiset final_conf = NFA::final_config(input_str);
    return in(final_conf, state);
};

bool NFA::accepts(const deque<int> &input_str)
{
    uiset final_conf = NFA::final_config(input_str);
    return not_empty_intersection(final_conf, _final_states);
}

deque<int> int_to_binary(int number, int length)
{
    // Initialize result with `length` zeros
    deque<int> result(length);
    for (int bit_idx = 0; bit_idx < length; bit_idx++)
        result[length - bit_idx - 1] = (number >> bit_idx) & 1;
    return result;
}

long long NFA::bruteforce_count_only(int n)
{
    long long count = 0;
    long long max_string_id = 1 << n;
    long long string_id;
    deque<int> string;
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
            // Compute the new final states if needed
            if (layer == n && in(_final_states, state))
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
                for (int i = 0; i < n; i++)
                {
                    // Insert the [(p, i)][a] -> (q, i+1) into the new transitions
                    bool valid = (old_state_layer_to_new_states.find({p, i}) !=
                                  old_state_layer_to_new_states.end());
                    if (!valid)
                        continue;
                    int new_p_layer_i = old_state_layer_to_new_states[{p, i}];
                    int new_q_layer_i_1 = old_state_layer_to_new_states[{q, i + 1}];
                    new_trans[new_p_layer_i][a].insert(new_q_layer_i_1);
                }
            }
        }
    }
    NFA new_nfa = NFA(new_states, _input_symbols, new_trans, new_init_states, new_final_states);
    for (auto new_state : new_nfa._states)
    {
        auto old_state_layer = new_state_to_old_state_layer[new_state];
        states_by_layer[old_state_layer.second].insert(new_state);
    }
    new_nfa._states_by_layer = states_by_layer;
    new_nfa._pre_unroll_state_map = new_state_to_old_state_layer;
    return new_nfa;
}

double NFA::compute_n_for_single_state(int state)
{
    if (in(_n_for_states, state))
        return _n_for_states[state];
    // will add the N(Pᵅ) calculated with `compute_n_for_states_set`
    double n_q_alpha = 0;
    if (!in(_reverse_transitions, state))
    {
        _n_for_states[state] = 0;
        return 0;
    }
    for (auto &[a, leading_states] : _reverse_transitions[state])
    {
        n_q_alpha += compute_n_for_states_set(leading_states);
    }
    // Cache the result for later
    _n_for_states[state] = n_q_alpha;
    return n_q_alpha;
}

double NFA::compute_n_for_states_set(uiset &states)
{
    int len_states = states.size();
    if (len_states == 0)
        return 0.0;
    // isets are hashable, while uisets aren't
    // we need to cache them, so the key is an iset
    iset ordered_states;
    ordered_states.insert(states.begin(), states.end());
    if (in(_n_for_sets, ordered_states))
        return _n_for_sets[ordered_states];
    // linear order ≺
    vector<int> states_list(ordered_states.begin(), ordered_states.end());
    // Count the first state in the list alone
    double intersection_rate, total = _n_for_states[states_list[0]];
    for (int i = 1; i < len_states; i++)
    {
        // states[i] = (q, i), where q is the original state name in A
        int anchor_state = states_list[i];
        int s_size = 0;
        double intersection_count = 0;
        // Now estimate the intersection rate for anchor_state
        for (auto &[string, count] : _s_for_states[anchor_state])
        {
            s_size += count;
            bool was_reachable = false;
            for (int j = 0; j < i; j++)
            {
                int previous_state = states_list[j];
                if (reachable(string, previous_state))
                {
                    was_reachable = true;
                    break;
                }
            }
            // Whether string is not in L(q_i) for every q_i < anchor
            if (!was_reachable)
                intersection_count += count;
        }
        intersection_rate = s_size > 0 ? intersection_count / s_size : 0.0;
        total += compute_n_for_single_state(anchor_state) * intersection_rate;
    }
    // Cache the result for later
    _n_for_sets[ordered_states] = total;
    return total;
}

deque<int> NFA::sample(int beta, uiset &states, deque<int> &curr_string, float phi, float phi_multiple)
{

    if (beta == 0)
    {
        if (rand_zero_one(gen) <= phi * phi_multiple)
            return curr_string;
        return {};
    }
    // p_beta_b will store all the states on layer i-1
    // which lead to r after reading b={0,1}
    // {'0': {...leading_states}, '1': {...leading_states}}
    umap<int, uiset> p_beta_b;
    // n_p_beta_b will store all the N(Pᵅ) calculated with
    // `compute_n_for_states_set`
    umap<int, int> n_p_beta_b;
    for (auto b : _sorted_symbols)
    {
        // Fill the leading states to do the backward pass
        uiset all_leading_states, leading_states;
        for (auto r : states)
        {
            if (in(_reverse_transitions, r) && in(_reverse_transitions[r], b))
            {
                leading_states = _reverse_transitions[r][b];
                all_leading_states.insert(leading_states.begin(), leading_states.end());
            }
        }
        p_beta_b[b] = all_leading_states;
        n_p_beta_b[b] = all_leading_states.size() == 0 ? 0 : compute_n_for_states_set(all_leading_states);
    }
    double sum_n_p_beta = 0;
    for (auto &[b, n_p_beta] : n_p_beta_b)
        sum_n_p_beta += n_p_beta;
    if (sum_n_p_beta == 0)
        return {};
    // Now that we have the sums for each leading symbol,
    // compute the weights to sample the next one.
    int n_symbols = _sorted_symbols.size();
    vector<double> weights(n_symbols);
    for (int s = 0; s < n_symbols; s++)
        weights[s] = n_p_beta_b[_sorted_symbols[s]] / sum_n_p_beta;

    discrete_distribution<int> weighted_sampler(weights.begin(), weights.end());
    int chosen_symbol = _sorted_symbols[weighted_sampler(gen)];
    sampled_symbols++;
    // w_beta-1 = b · w_beta
    curr_string.insert(curr_string.begin(), chosen_symbol);
    // p_beta-1
    uiset chosen_states = p_beta_b[chosen_symbol];
    //  phi / p_b
    float new_probability = phi / (n_p_beta_b[chosen_symbol] / sum_n_p_beta);
    return sample(beta - 1, chosen_states, curr_string, new_probability, phi_multiple);
}
// Computes c(κ)
ll retries_per_sample(ll kappa)
{
    return ceil(
        (2 + log(4) + 8 * log(kappa)) / log(1.0 / (1.0 - exp(-9))));
}

double NFA::count_accepted(int n, float epsilon, int kappa_multiple, float phi_multiple)
{
    if (!_states.size())
    {
        cout << "Empty NFA\n";
        return 0;
    }
    ll kappa = ceil(n * _states.size() / epsilon);
    // c(κ)
    ll retries_sample = retries_per_sample(kappa);
    cout << "retries_per_sample " << retries_sample << endl;
    ll sample_size = kappa_multiple * kappa;
    ll sample_hits = 0, sample_misses = 0;

    cout << "sample_size " << sample_size << endl;
    double exp_minus_five = exp(-5);
    // For each state q ∈ I, set N(q_0) = |L(q_0)| = 1
    // and S(q_0) = L(q_0) = {λ}
    deque<int> empty_vector;
    map<deque<int>, ll> empty_string;
    empty_string[empty_vector] = sample_size;
    for (auto q : _states_by_layer[0])
    {
        // The empty string will have sample_size samples
        _s_for_states[q] = empty_string;
        // And there's only one way to accept the empty string
        _n_for_states[q] = 1.0;
    }
    // For each i = 1, . . . , n and state q ∈ Q:
    //   (a) Compute N(q_i) given sketch[i-1]
    //   (b) Sample polynomially many uniform elements from L(q_i) using
    //       N(q_i) and sketch[i-1], and let S(q_i) be the multiset of
    //       uniform samples obtained
    double n_q_alpha;
    for (int i = 1; i <= n; i++)
    {
        for (auto q : _states_by_layer[i])
        {
            n_q_alpha = compute_n_for_single_state(q);
            if (n_q_alpha == 0)
            {
                cout << "Got 0 when computing n_q_alpha for q=" << q << endl;
                exit(1);
            }
            map<deque<int>, ll> this_q_samples;
            // sample probability
            float phi = exp_minus_five / n_q_alpha;
            for (ll sample_i = 0; sample_i < sample_size; sample_i++)
            {
                bool sampled_successfully = false;
                for (ll retry = 0; retry < retries_sample; retry++)
                {
                    uiset sampler_states = {q};
                    deque<int> sampler_empty = {};
                    deque<int> potential_sample = sample(i, sampler_states, sampler_empty, phi, phi_multiple);
                    if (!potential_sample.size())
                    {
                        sample_misses++;
                        continue;
                    }
                    // update this_q_samples' count with
                    // the just sampled string
                    if (!in(this_q_samples, potential_sample))
                        this_q_samples[potential_sample] = 1;
                    else
                        this_q_samples[potential_sample]++;
                    sampled_successfully = true;
                    sample_hits++;
                    break;
                }
                if (!sampled_successfully)
                    return 0;
            }
            _s_for_states[q] = this_q_samples;
        }
    }
    cout << "sample_misses " << sample_misses << "\n"
         << "sample_hits " << sample_hits << "\n"
         << "sampled_symbols " << sampled_symbols << "\n"
         << "miss_ratio " << (double)sample_misses / sample_hits << endl;
    // |L(F^n)|
    return compute_n_for_states_set(_final_states);
}