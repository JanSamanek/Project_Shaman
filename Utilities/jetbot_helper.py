def get_motor_speed(offset, saturation=0.15, turn_gain=0.4):
    if offset is not None:
        mot_speed_1, mot_speed_2 = (turn_gain * offset, -turn_gain * offset)
        if abs(mot_speed_1) >= saturation or abs(mot_speed_2) >= saturation:
            mot_speed_1, mot_speed_2 = saturation, -saturation
    else:
        mot_speed_1, mot_speed_2 = None, None
    return mot_speed_1, mot_speed_2