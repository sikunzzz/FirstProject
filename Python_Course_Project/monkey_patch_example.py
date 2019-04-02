from myfunc2 import run_func


def sub_tract(x, y):
    return x-y


def test_func(monkeypatch):
    monkeypatch.setattr("myfunc2.add_two", sub_tract)
    x = run_func(3, 5)
    assert x == -2