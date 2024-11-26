def projected_cash_flow_recursive(initial_cash, growth_rate, periods):
    if periods == 0:
        return [initial_cash]
    next_cash = initial_cash * (1 + growth_rate)
    return [initial_cash] + projected_cash_flow_recursive(next_cash, growth_rate, periods - 1)

print(projected_cash_flow_recursive(100, 0.1, 5))



