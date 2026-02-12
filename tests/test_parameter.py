def test_parameter_creation():
    from djinn.core.strategy.parameter import Parameter, param

    p = param(default=10, min=2, max=100, description="Test param")

    assert p.default == 10
    assert p.min == 2
    assert p.max == 100
    assert p.description == "Test param"
