#include <bits/stdc++.h>

using namespace std;

int in(unordered_set<int> set_to_search, int value)
{
    return set_to_search.find(value) != set_to_search.end();
}
int in(set<int> set_to_search, int value)
{
    return set_to_search.find(value) != set_to_search.end();
}