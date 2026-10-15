from paz.layers import RMSNormalization


def RMSNorm(epsilon=1e-5, name=None):
    return RMSNormalization(epsilon=epsilon, name=name)
