#include <bits/stdc++.h>

using namespace std;

template <class T, class U>
int in(T where_to_search, U value)
{
    return where_to_search.find(value) != where_to_search.end();
}

template <class T, class U>
T intersect(T set_1, U set_2)
{
    int smaller = set_1.size() < set_2.size() ? 1 : 2;
    auto smaller_set = smaller == 1 ? set_1 : set_2;
    auto bigger_set = smaller == 1 ? set_2 : set_1;
    T inter;
    for (auto e : smaller_set)
    {
        if (in(bigger_set, e))
            inter.insert(e);
    }
    return inter;
}