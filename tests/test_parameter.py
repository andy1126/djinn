def test_parameter_creation():
    from djinn.core.strategy.parameter import Parameter, param

    p = param(default=10, min=2, max=100, description="Test param")

    assert p.default == 10
    assert p.min == 2
    assert p.max == 100
    assert p.description == "Test param"


def test_parameter_validation():
    from djinn.core.strategy.parameter import Parameter, param

    p = param(default=10, min=2, max=100)

    # 有效值
    assert p.validate(50) == 50

    # 边界值
    assert p.validate(2) == 2
    assert p.validate(100) == 100

    # 无效值应该抛出异常
    try:
        p.validate(1)  # 低于最小值
        assert False, "应该抛出 ValueError"
    except ValueError:
        pass

    try:
        p.validate(101)  # 高于最大值
        assert False, "应该抛出 ValueError"
    except ValueError:
        pass
