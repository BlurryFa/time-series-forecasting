def g_f_filter(data, x ,dx, g, h, dt=1):

    begining = x
    estimates = []

    for measurement in data:
        x += dx * dt
        dx = dx
        redidual = measurement - x
        estimate = x + redidual * g
        dx = dx + h * (redidual) / dt
        estimates.append(estimate)

    return estimates



