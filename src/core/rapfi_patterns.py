from enum import IntEnum

class RapfiPattern(IntEnum):
    """
    Mapping for Rapfi's Pattern4 internal codes.
    Source: Rapfi/core/types.h
    """
    NONE = 0            # Anything else
    FORBID = 1          # Forbidden point (Renju)
    L_FLEX2 = 2         # F2 + Any
    K_BLOCK3 = 3        # B3 + Any
    J_FLEX2_2X = 4      # F2 x 2
    I_BLOCK3_PLUS = 5   # B3 x 2 | B3 + F2
    H_FLEX3 = 6         # F3 + Any
    G_FLEX3_PLUS = 7    # F3 + F2 | F3 + B3
    F_FLEX3_2X = 8      # F3 x 2
    E_BLOCK4 = 9        # B4 + Any
    D_BLOCK4_PLUS = 10  # B4 + F2 | B4 + B3
    C_BLOCK4_FLEX3 = 11 # B4 + F3
    B_FLEX4 = 12        # F4 | F4S | B4 x 2
    A_FIVE = 13         # F5

    @classmethod
    def describe(cls, code: int) -> str:
        """Returns a human-readable description of the pattern."""
        try:
            return cls(code).name
        except ValueError:
            return f"UNKNOWN({code})"
