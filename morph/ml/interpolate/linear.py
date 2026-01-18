def linear_interpolate(z1, z2, steps):
    """
    Simple linear interpolation in latent space
    """
    interpolated = []

    for i in range(steps):
        alpha = i / (steps - 1)
        z = (1 - alpha) * z1 + alpha * z2
        interpolated.append(z)

    return interpolated
