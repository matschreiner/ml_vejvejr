import os

import numpy as np


def main():
    temp_profile = np.random.rand(100, 10)
    W = np.random.randn(10, 10)
    temp_profile_delta = temp_profile.dot(W) + 0.1 * np.random.randn(100, 10)

    os.makedirs("data", exist_ok=True)
    np.savez(
        "data/test_data.npz",
        input=temp_profile,
        target=temp_profile_delta,
    )


main()
