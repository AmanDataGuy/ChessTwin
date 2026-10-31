"""Stage 0 sanity check: proves pytest + CI are wired up before any real modules exist."""


def test_sanity():
    assert 1 + 1 == 2
