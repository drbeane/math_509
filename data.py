def generate_values():
    import numpy as np
    from sklearn.datasets import make_classification  
  
    N = 1000
    np.random.seed(1)
    x1 = np.random.normal(500, 100, N).astype(int)
    np.random.seed(597474)
    ep = np.random.normal(0, 0.75, N)
    y1 = 16 * (x1/1000) ** 5.8 * np.exp(ep)
    y1 = y1.round(4)
        
    X2, y2 = make_classification(n_samples=1000, n_classes=3, n_features=100, n_informative=20, 
                           n_redundant=10, class_sep=1.6, random_state=16)

    np.random.seed(1)
    X3 = np.random.normal(100, 20, size=(100,5))
    betas = np.array([15, 32, 4.5, -27, -17.5])
    y3 = X3 @ betas + np.random.normal(0, 400, 100)
    X3_ones = np.hstack([np.ones((100,1)), X3])

    return x1, y1, X2, y2, X3, y3, X3_ones

x1, y1, X2, y2, X3, y3, X3_ones = generate_values()
