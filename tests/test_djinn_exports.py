def test_djinn_package_exports():
    """测试关键类可以直接从 djinn 包导入"""
    import djinn

    # 验证关键导出存在
    assert hasattr(djinn, 'SimpleStrategy')
    assert hasattr(djinn, 'param')
    assert hasattr(djinn, 'Parameter')
