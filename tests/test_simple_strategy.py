def test_simple_strategy_parameter_collection():
    from djinn.core.strategy.simple import SimpleStrategy
    from djinn.core.strategy.parameter import param

    class TestStrategy(SimpleStrategy):
        fast = param(10, min=2, max=100)
        slow = param(30, min=5, max=200)

    strategy = TestStrategy()

    assert strategy.params.fast == 10
    assert strategy.params.slow == 30
