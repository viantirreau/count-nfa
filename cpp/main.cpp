#include <bits/stdc++.h>
#include "utils.cpp"
#include "nfa.hpp"

using namespace std;
using namespace std::chrono;

int main(int argc, char **argv)
{
	bool calc_bruteforce = true;
	if (argc > 1 && strcmp(argv[1], "0") == 0)
		calc_bruteforce = false;

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
	auto t0 = high_resolution_clock::now();
	double estimation = unrolled.count_accepted(n_str_len, epsilon, kappa_multiple, phi_multiple);
	auto t1 = high_resolution_clock::now();
	if (calc_bruteforce)
	{
		ll bruteforce = nfa.bruteforce_count_only(n_str_len);
		auto t2 = high_resolution_clock::now();
		auto bruteforce_time = duration_cast<milliseconds>(t2 - t1).count();
		std::cout << "bruteforce " << bruteforce << "\n"
				  << "bruteforce_time " << bruteforce_time << endl;
	}
	auto estimation_time = duration_cast<milliseconds>(t1 - t0).count();

	std::cout << "estimation " << (int)estimation << "\n"
			  << "estimation_time " << estimation_time << endl;

	return 0;
}
