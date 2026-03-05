# Protocol & Models

The protocol layer defines the data structures used to communicate with engines.

## Core Models

### Move

The `Move` class represents a board coordinate. It is the common currency of the API.

```python
from pygomo import Move

# Creating moves
m1 = Move("h8")      # Algebraic
m2 = Move((7, 7))    # Tuple
m3 = Move("7,7")     # Numeric

# Properties
print(m1.col)        # 7
print(m1.row)        # 7
print(str(m1))       # "h8"
```

### SearchInfo

Parsing engine output is complex. `SearchInfo` encapsulates the realtime data stream.

*   `depth`: Current search depth.
*   `eval`: A `Evaluate` object containing the raw score.
*   `winrate`: Calculated winrate [0.0 - 1.0].
*   `pv`: Principal Variation (list of `Move`).
*   `time_ms`: Time spent searching.
*   `nodes`: Total nodes visited.

### Evaluate & Winrate

Engine scores are typically in "centipawns" relative to Black.
*   Positive (+): Black advantage.
*   Negative (-): White advantage.
*   `winrate_percent()`: Converts this score to a human-readable percentage [0-100%].

## Class Reference

```{eval-rst}
.. autoclass:: pygomo.protocol.models.Move
    :members:
    :undoc-members:

.. autoclass:: pygomo.protocol.models.SearchInfo
    :members:
    :undoc-members:

.. autoclass:: pygomo.protocol.models.Evaluate
    :members: winrate, winrate_percent, score
    :undoc-members:
```
