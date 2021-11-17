#include <bits/stdc++.h>
using namespace std;

#define uset unordered_set
#define umap unordered_map
class NFA
{
private:
    /* data */
public:
    uset<int> states;
    uset<int> input_symbols;
    uset<int> initial_states;
    uset<int> final_states;
    umap<int, map<int, set<int>>> transitions;
    NFA(/* args */);
    ~NFA();
};
