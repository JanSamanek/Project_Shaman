def get_motor_speed(offset, saturation=0.1, turn_gain=0.3):
    if offset is not None:
        mot_speed_1, mot_speed_2 = (turn_gain * offset, -turn_gain * offset)
        if abs(mot_speed_1) >= saturation or abs(mot_speed_2) >= saturation:
            mot_speed_1, mot_speed_2 = _sign(offset)*saturation, -_sign(offset)*saturation
    else:
        mot_speed_1, mot_speed_2 = None, None
    return mot_speed_1, mot_speed_2


def _sign(x):
    if x == 0:
        return 0
    elif x < 0:
        return -1
    else:
        return 1