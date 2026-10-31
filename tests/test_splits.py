from data.splits import split_games


def _synthetic_game_ids(n: int) -> list[str]:
    return [f"game_{i}" for i in range(n)]


def test_split_covers_every_game_exactly_once():
    game_ids = _synthetic_game_ids(1000)
    assignment = split_games(game_ids)

    assert set(assignment.keys()) == set(game_ids)
    assert all(split in ("train", "val", "test") for split in assignment.values())


def test_split_ratio_is_approximately_80_10_10():
    game_ids = _synthetic_game_ids(1000)
    assignment = split_games(game_ids)

    counts = {"train": 0, "val": 0, "test": 0}
    for split in assignment.values():
        counts[split] += 1

    assert counts["train"] == 800
    assert counts["val"] == 100
    assert counts["test"] == 100


def test_split_is_reproducible():
    game_ids = _synthetic_game_ids(500)
    first = split_games(game_ids)
    second = split_games(game_ids)
    assert first == second


def test_split_is_order_independent():
    game_ids = _synthetic_game_ids(500)
    shuffled = list(reversed(game_ids))
    assert split_games(game_ids) == split_games(shuffled)


def test_no_game_appears_in_two_splits():
    game_ids = _synthetic_game_ids(2000)
    assignment = split_games(game_ids)
    # Every game_id is a unique dict key by construction, but confirm the
    # underlying invariant explicitly: exactly one split per game.
    assert len(assignment) == len(set(game_ids))
