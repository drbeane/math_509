def generate_values():
    import numpy as np

    ##########################################
    # Generate Naive Bayes Data
    ##########################################
    np.random.seed(6)
    X1 = np.random.choice([0,1], size=(100_000,10), p=[0.8, 0.2])
    betas = np.random.uniform(0, 1, size=10)
    z = X1 @ betas - 1
    p = 1 / (1 + np.exp(-z))
    y1 = np.where(p < 0.5, 0, 1)

    print((y1 == 1).mean())

    Nspam = 462 + 500
    Nham = 10000 - Nspam

    s0 = np.argwhere(y1 == 0)[:Nham].flatten()
    s1 = np.argwhere(y1 == 1)[:Nspam].flatten()
    sel = np.hstack([s0,s1])
    sel = np.random.choice(sel, size=10_000, replace=False)

    X1 = X1[sel, :]
    y1 = y1[sel]

  
    return X1, y1, None, None

X1, y1, X2, y2 = generate_values()
