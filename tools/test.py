import numpy as np
def _round_to_staff_pos(x: float):
    rounded = np.round(x)
    even = (rounded + 2000) % 2 == 0
    if not even:
        if abs(x - rounded) < 0.45:
            return rounded
        else:
            return rounded + 1 if x - rounded > 0 else rounded - 1
    else:
        return rounded

for x in range(0,20):
    x = x / 20
    print(x)
    print( f'staff {_round_to_staff_pos(x*2)}')