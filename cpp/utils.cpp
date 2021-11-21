#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Implements an equivalent of Python `in`
 * keyword for checking set/map/iterator pertenence.
 * 
 * The runtime depends on how fast std::find is in
 * this data structure. 
 * 
 * @param where_to_search: iterable
 * @param value
 * @return int 
 */
template <class T, class U>
bool in(T where_to_search, U value)
{
    return where_to_search.find(value) != where_to_search.end();
}

/**
 * @brief Implements a generic set intersection.
 * Supports unordered_set of any type as well.
 *  
 * @tparam T 
 * @tparam U 
 * @param set_1 
 * @param set_2 
 * @return T 
 */
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

/**
 * @brief Implements a generic set intersection.
 * Returns true if the intersection is not empty.
 *  
 * @tparam T 
 * @tparam U 
 * @param set_1 
 * @param set_2 
 * @return true | false 
 */
template <class T, class U>
bool not_empty_intersection(T set_1, U set_2)
{
    int smaller = set_1.size() < set_2.size() ? 1 : 2;
    auto smaller_set = smaller == 1 ? set_1 : set_2;
    auto bigger_set = smaller == 1 ? set_2 : set_1;
    for (auto e : smaller_set)
    {
        if (in(bigger_set, e))
            return true;
    }
    return false;
}
