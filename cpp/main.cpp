#include <bits/stdc++.h>
#include "utils.cpp"
#include "nfa.hpp"

using namespace std;

int main(int argc, char **argv)
{
	// fast I/O
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);
	int n_states, n_symbols, n_trans, n_str_len, kappa_multiple;
	float epsilon, phi_multiple;
	// empty set container
	unordered_set<int> states;
	vector<int> ordered_states;
	unordered_set<int> initial_states;
	unordered_set<int> final_states;
	unordered_map<int, map<int, unordered_set<int>>> transitions;
	unordered_set<int> input_symbols;

	cin >> n_str_len >> epsilon >> kappa_multiple >> phi_multiple;
	cin >> n_states >> n_symbols >> n_trans;

	// Read state labels
	int placeholder;
	for (int i = 0; i < n_states; i++)
	{
		cin >> placeholder;
		states.insert(placeholder);
		ordered_states.push_back(placeholder);
	}
	// Read initial states
	for (int i = 0; i < n_states; i++)
	{
		cin >> placeholder;
		if (placeholder) // was a one
			initial_states.insert(ordered_states[i]);
	}
	// Read final states
	for (int i = 0; i < n_states; i++)
	{
		cin >> placeholder;
		if (placeholder) // was a one
			final_states.insert(ordered_states[i]);
	}
	// Read input symbols
	for (int i = 0; i < n_symbols; i++)
	{
		cin >> placeholder;
		input_symbols.insert(placeholder);
	}
	// Read transitions
	int p, a, q;
	for (int i = 0; i < n_trans; i++)
	{
		cin >> p >> a >> q;
		transitions[p][a].insert(q);
	}

	NFA nfa = NFA(states, input_symbols, transitions, initial_states, final_states);
	NFA unrolled = nfa.unroll(n_str_len);
	double estimation = unrolled.count_accepted(n_str_len, epsilon, kappa_multiple, phi_multiple);
	std::cout << "Bruteforce: " << nfa.bruteforce_count_only(n_str_len) << "\n"
			  << "Estimation: " << estimation << endl;

	return 0;
}
