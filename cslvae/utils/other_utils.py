from typing import Any, List


def flatten_list(list_of_lists: List[List[Any]]) -> List[Any]:
    """Flattens a list of lists into a single list."""
    return [item for sublist in list_of_lists for item in sublist]


def int2mix(number: int, radix: List[int]) -> List[int]:
    assert isinstance(radix, list) and all(isinstance(i, int) for i in radix)
    mix = []
    radix_rev = radix[::-1]
    for i in range(0, len(radix_rev)):
        mix.append(number % radix_rev[i])
        number //= radix_rev[i]
    if number > 0:
        raise ValueError
    mix.reverse()
    return mix
