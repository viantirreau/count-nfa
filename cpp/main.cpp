#include <iostream>
#include <iterator>
#include <set>
#include <map>
#include <algorithm>
#include <vector>
#include <queue>
#include <utility>

using namespace std;

int in(set<int> set_to_search, int value)
{
	return set_to_search.find(value) != set_to_search.end();
}

int main(int argc, char **argv)
{

	// empty set container
	set<int> initial_states;
	set<int> final_states;
	map<int, map<int, set<int>>> transitions;
	if (argc < 2)
		exit(1);

	int n = atoi(argv[1]);

	initial_states.insert(0);
	final_states.insert({2, 6});
	std::vector<int> v_intersection;
	// transitions.insert(0, 0, 1);
	// transitions.insert(0, 1, 3);
	// transitions.insert(1, 0, 2);
	// transitions.insert(2, 0, 2);
	// transitions.insert(2, 1, 2);
	// transitions.insert(3, 0, 4);
	// transitions.insert(4, 0, 5);
	// transitions.insert(5, 0, 6);
	// transitions.insert(6, 0, 6);
	// transitions.insert(0, 1, 6);

	if (n == 0)
	{
		set_intersection(initial_states.begin(), initial_states.end(), final_states.begin(), final_states.end(), std::back_inserter(v_intersection));
		int inter_size = v_intersection.size();
		cout << (inter_size > 0 ? "1" : "0") << std::endl;
		exit(0);
	}

	int accepted_count = 0;
	deque<pair<int, int>> visit_queue;

	std::set<int>::iterator it = initial_states.begin();
	while (it != initial_states.end())
		visit_queue.emplace_back(pair<int, int>(*it++, 0));

	while (!visit_queue.empty())
	{
		pair<int, int> curr_state_len = visit_queue.back();
		visit_queue.pop_back();
		int curr_state = std::get<0>(curr_state_len);
		int curr_len = std::get<1>(curr_state_len);
		if (curr_len == n)
		{
			if (in(final_states, curr_state))
			{
				accepted_count++;
			}
			continue;
		}
	}
	cout << accepted_count << '\n';

	cout << "2 in final_states " << in(final_states, 2) << std::endl;

	return 0;
}
