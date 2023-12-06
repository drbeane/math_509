def generate_values():
    import numpy as np
    from sklearn.datasets import make_classification

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

    ##########################################
    # Generate Logistic Regression Data
    ##########################################
    X2, y2 = make_classification(n_samples=1000, n_features=4, n_informative=4, n_redundant=0, n_classes=2, random_state=7)

    
    return X1, y1, X2.round(3), y2

def unit_test_1():
    global X2 
    global add_ones

    XE = add_ones(X2)
    
    try:
        XE = add_ones(X2)
    except:
        print('Function results in an error.')
        return
    
    if not isinstance(XE, np.ndarray):
        print(f'Returned object is not an array. It has type {type(XE)}.')
        return 

    try:
        if XE.shape != (1000, 5):
            print('Shape of returned array is not correct.')
            return
    except:
        print('Unable to determine array shape.')
        return

    if (XE[:,0] == 1).sum() != 1000:
        print('First column does not consist entirely of ones.')
        return
    
    if np.sum(XE[:,1:] == X2) != 4000:
        print('Returned array does not contain copy of feature array.')
        return
    
    print('Tests passed.')


X1, y1, X2, y2 = generate_values()
