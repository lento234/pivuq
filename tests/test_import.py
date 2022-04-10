from importlib import metadata

def test_import():
    import pivuq
    assert pivuq.__version__ == metadata.version("pivuq")