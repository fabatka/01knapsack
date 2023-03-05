from collections import Counter
from functools import reduce
from itertools import combinations
from operator import add


def maximum_value(orders, maximum_weight):
    """Returns the maximum value of the orders we can group together respecting the given weight limit

    :param orders: list of orders, that must be dictionaries with 'value' and 'weight' keys
    :param maximum_weight: the maximum weight that the batch of orders can't exceed
    :return: The maximum possible value of the batch of orders that respect the maximum weight limit
    """
    # if we had to deal with large number of orders, we could use approximation methods above the recursion limit
    # if len(orders) > 990:
    #     return maximum_value_greedy_approx(orders, maximum_weight)
    return maximum_value_dynamic_solution_memoized(orders, maximum_weight)


# for the test cases it's fine, but after about 20-30 elements it slows down tremendously, as expected
def maximum_value_naive(orders, maximum_weight):
    """The naive approach is to simply iterate over every possible combination of orders"""
    # remove orders that are too big by themselves
    max_payload_value = 0
    orders_filtered = [order for order in orders if not order['weight'] > maximum_weight]
    for length in range(len(orders_filtered), 0, -1):
        for combination in combinations(orders_filtered, length):
            payload = reduce(add, map(Counter, combination))
            if payload['weight'] <= maximum_weight and payload['value'] > max_payload_value:
                max_payload_value = payload['value']
    return max_payload_value


# one of the simpler textbook solutions that is limited by the recursion depth mostly (fine until 1000 elements)
def maximum_value_dynamic_solution_memoized(orders, maximum_weight):
    """A dynamic programming solution to the 0-1 knapsack problem, as seen on Wikipedia
    source:
    Knapsack problem (2023) Wikipedia. Wikimedia Foundation.
    Available at: https://en.wikipedia.org/wiki/Knapsack_problem (Accessed: March 5, 2023).

    The code is intentionally kept as similar as possible to the pseudocode about the memoized version of the dynamic
    programming approach
    """
    # introducing aliases to agree with the source's nomenclature
    n = len(orders)
    W = maximum_weight

    def w(i):
        return orders[i-1]['weight']

    def v(i):
        return orders[i-1]['value']

    value = [[-1 for _ in range(W + 1)] for _ in range(n + 1)]

    def m(i, j):
        """The maximum value we can get under the condition: use first i items, total weight limit is j"""
        if i == 0 or j <= 0:
            value[i][j] = 0
            return

        if value[i-1][j] == -1:
            m(i-1, j)

        if w(i) > j:
            value[i][j] = value[i-1][j]
        else:
            if value[i-1][j-w(i)] == -1:
                m(i-1, j-w(i))
            value[i][j] = max(value[i-1][j], value[i-1][j-w(i)] + v(i))

    m(n, W)

    return value[n][W]


# for large number of orders, it can be more practical to use approximative methods
# there exist better approximation algorithms to this problem, this is what I came up with myself
# this solution doesn't pass some tests, but those don't cover the use case for this function (+1000 orders)
def maximum_value_greedy_approx(orders, maximum_weight):
    """Perhaps the simplest greedy approximation algorithm for the problem
    We sort the orders in descending order by value/weight ratio, and try to fill the weight limit
    """
    max_payload_value = 0

    # remove orders that are too big by themselves
    filtered_orders = [order for order in orders if not order['weight'] > maximum_weight]
    if len(filtered_orders) == 0:
        return max_payload_value

    # calculate the value/weight ratio for all orders
    [order.update(ratio=order['value']/order['weight']) for order in filtered_orders]

    # sort the orders by this ratio in descending order
    sorted_filtered_orders = sorted(filtered_orders, key=lambda o: o['ratio'], reverse=True)

    # as we filtered out the too heavy orders, we can be sure that at least one order fits into the final payload
    final_order = [sorted_filtered_orders[0]]
    payload = reduce(add, map(Counter, final_order))
    for order in sorted_filtered_orders[1:]:
        if payload['weight'] == maximum_weight:
            break
        if payload['weight'] + order['weight'] > maximum_weight:
            continue
        final_order.append(order)
        payload = reduce(add, map(Counter, final_order))
    return payload['value']
