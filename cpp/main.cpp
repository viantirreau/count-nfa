#include <bits/stdc++.h>
#include "utils.cpp"
#include "nfa.hpp"

using namespace std;

int main(int argc, char **argv)
{

	// empty set container
	unordered_set<int> states = {0, 1};
	// unordered_set<int> states = {0, 1, 2, 3, 4, 5, 6};
	unordered_set<int> initial_states;
	unordered_set<int> final_states;
	unordered_map<int, map<int, unordered_set<int>>> transitions;
	unordered_set<int> input_symbols = {0, 1};
	initial_states.insert(0);
	final_states.insert(1);
	vector<int> v_intersection;

	transitions[0][0].insert({0, 1});
	transitions[1][1].insert(0);

	// transitions[0][0].insert(1);
	// transitions[0][1].insert(3);
	// transitions[1][0].insert(2);
	// transitions[2][0].insert(2);
	// transitions[2][1].insert(2);
	// transitions[3][0].insert(4);
	// transitions[4][0].insert(5);
	// transitions[5][0].insert(6);
	// transitions[6][0].insert(6);
	// transitions[6][1].insert(6);

	NFA nfa = NFA(states, input_symbols, transitions, initial_states, final_states);
	int n = atoi(argv[1]);
	// int n = 3;
	NFA unrolled = nfa.unroll(n);
	double estimation = unrolled.count_accepted(n, 1.0, 1.0);
	std::cout << "Bruteforce: " << nfa.bruteforce_count_only(n) << " | "
			  << "Estimation: " << estimation << endl;

	return 0;
}
