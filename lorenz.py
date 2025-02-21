import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time



def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

if input("Do you want to generate new Lorenz series (y)/n?\n") in {'y', ''}:
    length = int(input("How long should the series be?\nEnter a number from  to \n"))
    gamma = float("0."+input("Enter float from 0 to 1 \n'frequency of the saved points from the series'\n0."))
    n_points = max(2, int(200000 * gamma))

    sigma = 10
    rho = 28
    beta = 8 / 3

    initial_state = [1.0, 0.0, 0.0]
    t_span = (0, length)
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    solution = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval, args=(sigma, rho, beta))

    unix_time = np.arange(len(solution.t))
    df = pd.DataFrame({
        'Unix_Time': unix_time,
        'X': solution.y[0],
        'Y': solution.y[1],
        'Z': solution.y[2]
    })

    df.drop(["Y", "Z"], axis=1, inplace=True)
    df.to_csv('lorenz_attractor.csv', index=False) 
else:
    df = pd.read_csv("lorenz_attractor.csv")

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(df.index, df['X'], color='b')
ax.set_xlabel('Index')
ax.set_ylabel('X')
plt.title('Lorenz Attractor')
plt.show()
