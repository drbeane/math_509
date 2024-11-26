def gradient_optimization(sym_fn, vars, init_vals, alpha, max_iter,
                          threshold, mode='min', verbosity=0):
    import sympy as sym
    import numpy as np

    if mode not in ['max', 'min']:
        raise Exception("The mode parameter must be set to 'max' or 'min'.")
                            
    gradient = sym.derive_by_array(sym_fn, vars)

    params = np.array(init_vals)
    param_list = [params]

    sub_dict = {var:val for var, val in zip(vars, params)}
    f_val = sym_fn.subs(sub_dict)

    found = False

    for i in range(1, max_iter+1):
        grad = gradient.subs(sub_dict)

        if mode == 'min':
            params = params - alpha * grad
        elif mode == 'max':
            params = params + alpha * grad

        param_list.append(params)

        old_f_val = f_val
        sub_dict = {var:val for var, val in zip(vars, params)}
        f_val = sym_fn.subs(sub_dict)

        if (verbosity > 0) and (i % verbosity == 0):
            param_str = ', '.join([f'{p:.4f}' for p in params])
            print(f'Iteration {i:03}:  params = [{param_str}],  f(params) = {f_val:.4f}')

        if abs(old_f_val - f_val) < threshold:
            found = True
            break

        if abs(f_val) > 1e20:
            break

    if verbosity > 0: print()
    if found:

        print(f'Algorithm terminated after {i} steps.')
        param_str = ', '.join([f'{p:.4f}' for p in params])
        print(f'Final parameter estimates: [{param_str}]')
        print(f'Optimal value: {f_val:.4f}')
        return params, param_list

    else:
        print('Maximum iterations reached without algorithm terminating.')
        return None, param_list


def one_var_grad_plot(fn, vars, param_list, xlim, ylim, fs=[6,3], yscale=None):
    f_values = [fn(x[0]) for x in param_list]
    import numpy as np
    import matplotlib.pyplot as plt

    x_grid = np.linspace(xlim[0], xlim[1], 100)
    y_grid = fn(x_grid)

    plt.figure(figsize=fs)
    plt.plot(x_grid, y_grid)
    plt.plot(param_list, f_values, c='darkorange')
    plt.axhline(0, c='k', linewidth=0.5)
    plt.axvline(0, c='k', linewidth=0.5)
    plt.scatter(param_list, f_values, c='darkorange', s=10, zorder=2)
    plt.scatter(param_list[-1], f_values[-1], c='darkorange', edgecolor='k', s=40, zorder=2)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if yscale is not None:
        plt.yscale(yscale)

    plt.show()

def two_var_grad_plot(fn, vars, param_list, xlim, ylim, fs=[12,6], zoom=True):
    import numpy as np
    import matplotlib.pyplot as plt
  
    # x and y coords for points along gradient path
    x_path = [float(params[0]) for params in param_list]
    y_path = [float(params[1]) for params in param_list]

    # Get dimensions of window
    x_min, x_max = np.min(x_path), np.max(x_path)
    y_min, y_max = np.min(y_path), np.max(y_path)
    width = 1.1*np.max([x_max - x_min, y_max - y_min])
    win_x = [(x_min + x_max - width)/2, (x_min + x_max + width)/2]
    win_y = [(y_min + y_max - width)/2, (y_min + y_max + width)/2]

    # Grid points and heights for contour map
    N = 500
    (a, b), (c, d) = xlim, ylim
    dx, dy = b-a, d-c
    X = np.vstack([np.linspace(a-dx, b+dx, N)]*N)
    Y = np.vstack([np.linspace(c-dy, d+dy, N)]*N).T
    h = fn(X,Y)

    # Define Levels
    lmax = h.max(); lmin = h.min()
    lvls = np.linspace(lmin, lmax, 16)


    plt.figure(figsize=fs)

    # Large plot
    if zoom:
        plt.subplot(1, 2, 1)
    plt.contourf(X, Y, h, levels=lvls, cmap='RdBu')
    plt.contour(X, Y, h, levels=lvls, linestyles='solid', linewidths=1, colors='#777777', zorder=3)
    plt.plot(x_path, y_path, c='k', zorder=5)
    plt.scatter(x_path, y_path, c='k', zorder=5, s=5)
    plt.scatter(x_path, y_path, c='darkorange', zorder=4, s=15)
    plt.axis('On')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid()

    if zoom:
        # Window
        plt.subplot(1, 2, 2)
        plt.contourf(X, Y, h, levels=lvls, cmap='RdBu')
        plt.contour(X, Y, h, levels=lvls, linestyles='solid', linewidths=1, colors='#777777', zorder=3)
        plt.plot(x_path, y_path, c='k', zorder=5)
        plt.scatter(x_path, y_path, c='k', zorder=5, s=10)
        plt.scatter(x_path, y_path, c='darkorange', zorder=4, s=30)
        plt.axis('On')
        plt.xlim(win_x)
        plt.ylim(win_y)
        plt.grid()

    plt.show()
