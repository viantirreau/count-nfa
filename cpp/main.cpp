#include <bits/stdc++.h>
#include "utils.cpp"
#include "nfa.hpp"

using namespace std;

int main(int argc, char **argv)
{

	// empty set container
	unordered_set<int> initial_states;
	unordered_set<int> final_states;
	unordered_map<int, map<int, unordered_set<int>>> transitions;
	unordered_set<int> states = {0, 1, 2, 3, 4, 5, 6};
	unordered_set<int> input_symbols = {0, 1};
	initial_states.insert(0);
	final_states.insert({6});
	std::vector<int> v_intersection;

	transitions[0][0].insert(1);
	transitions[0][1].insert(3);
	// transitions[1][0].insert(2);
	transitions[2][0].insert(2);
	transitions[2][1].insert(2);
	transitions[3][0].insert(4);
	transitions[4][0].insert(5);
	transitions[5][0].insert(6);
	transitions[6][0].insert(6);
	transitions[6][1].insert(6);

	// if (n == 0)
	// {
	// 	unordered_set<int>::iterator it = initial_states.begin();
	// 	while (it != initial_states.end())
	// 	{
	// 		if (in(final_states, *it))
	// 		{
	// 			cout << "1" << std::endl;
	// 			exit(0);
	// 		}
	// 		it++;
	// 	}
	// 	cout << "0" << std::endl;
	// 	exit(0);
	// }

	NFA nfa = NFA(states, input_symbols, transitions, initial_states, final_states);
	NFA unrolled = nfa.unroll(100);
	// int accepted_count = 0;
	// deque<pair<int, int>> visit_queue;

	// std::unordered_set<int>::iterator it = initial_states.begin();
	// while (it != initial_states.end())
	// 	visit_queue.emplace_back(make_pair(*it++, 0));

	// while (!visit_queue.empty())
	// {
	// 	// read the element
	// 	pair<int, int> curr_state_len = visit_queue.back();
	// 	// delete it (returns void)
	// 	visit_queue.pop_back();
	// 	int curr_state = curr_state_len.first;
	// 	int curr_len = curr_state_len.second;
	// 	if (curr_len == n)
	// 	{
	// 		if (in(final_states, curr_state))
	// 		{
	// 			accepted_count++;
	// 		}
	// 		continue;
	// 	}
	// 	unordered_map<int, map<int, set<int>>>::iterator symbol_next_states = transitions.find(curr_state);
	// 	if (symbol_next_states != transitions.end())
	// 	{
	// 		for (auto symbol_next_state : symbol_next_states->second)
	// 		{
	// 			for (int next_state : symbol_next_state.second)
	// 			{
	// 				visit_queue.emplace_back(make_pair(next_state, curr_len + 1));
	// 			}
	// 		}
	// 	}
	// }

	// cout << accepted_count << '\n';

	// cout << "2 in final_states " << in(final_states, 2) << std::endl;

	return 0;
}
