import numpy as np
from scipy.spatial import distance


def pareto_frontier(xs, ys, maxy=False):
    """
    Method to take two equally-sized lists and return just the elements which lie
    on the Pareto frontier, sorted into order.
    Default behaviour is to find the maximum for both X and Y, but the option is
    available to specify maxX = False or maxY = False to find the minimum for either
    or both of the parameters.
    """
    # Sort the list in either ascending or descending order of X
    x_index = np.argsort(xs)
    my_list = [[xs[i], ys[i]] for i in x_index]

    # Start the Pareto frontier with the first value in the sorted list
    p_front = [my_list[0]]
    indices = [x_index[0]]
    # Loop through the sorted list
    for i, pair in enumerate(my_list[1:]):
        if maxy:
            if pair[1] >= p_front[-1][1]:  # Look for higher values of Y
                p_front.append(pair)       # and add them to the Pareto frontier
                indices += [x_index[i+1]]
        else:

            if pair[1] <= p_front[-1][1]:  # Look for lower values of Y
                p_front.append(pair)       # and add them to the Pareto frontier
                indices += [x_index[i+1]]

    # Turn resulting pairs back into a list of Xs and Ys
    p_frontx = [pair[0] for pair in p_front]
    p_fronty = [pair[1] for pair in p_front]
    return p_frontx, p_fronty, indices


def pareto_frontier1(Xs, Ys, maxX = False, maxY = False):
# Sort the list in either ascending or descending order of X
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
# Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]
# Loop through the sorted list
    for pair in myList[1:]:
        if maxY:
            if pair[1] >= p_front[-1][1]: # Look for higher values of Y
                p_front.append(pair)      #  and add them to the Pareto frontier
        else:
            if pair[1] <= p_front[-1][1]: # Look for lower values of Y
                p_front.append(pair)      #  and add them to the Pareto frontier

# Turn resulting pairs back into a list of Xs and Ys
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return p_frontX, p_frontY


def igd(params):
    obtained = params[0]
    ideals = params[1]
    if np.isnan(np.min(obtained)):
        return 10.0**6
    igd_val = 0
    maxdist = 10.0**10
    for d in ideals:
        min_dist = maxdist
        for o in obtained:
            min_dist = min(min_dist, distance.euclidean(o, d))
        igd_val += min_dist
    return igd_val/len(ideals), igd_val/np.sqrt(len(ideals))


class FFunctions:
    Obj_number = {"F1": 2, "F2": 2, "F3": 2, "F4": 2, "F5": 2, "F6": 3, "F7": 2, "F8": 2, "F9": 2}

    Dec_Var_ranges = {"F1": np.array([[0, 1], [0, 1], [0, 1]]), "F2": np.array([[0, 1], [-1, 1], [-1, 1]]),
                      "F3": np.array([[0, 1], [-1, 1], [-1, 1]]), "F4": np.array([[0, 1], [-1, 1], [-1, 1]]),
                      "F5": np.array([[0, 1], [-1, 1], [-1, 1]]), "F6": np.array([[0, 1], [0, 1], [-2, 2]]),
                      "F7": np.array([[0, 1], [0, 1], [0, 1]]), "F8": np.array([[0, 1], [0, 1], [0, 1]]),
                      "F9": np.array([[0, 1], [-1, 1], [-1, 1]])}

    def __init__(self, number_variables, function_name):
        self.n = number_variables
        self.fname = function_name
        self.n_obj = self.Obj_number[self.fname]
        self.DV_range = self.Dec_Var_ranges[self.fname]
        self.MOP_Function = self.MOP_Functions_Dir[self.fname]
        self.PS_Function = self.PS_Functions_Dir[self.fname]

    def f1_mop(self, x1):
        xvals = np.zeros((x1.shape[0], self.n))
        xvals[:, 0] = x1
        for j in range(2, self.n+1):
            aux = 0.5*(1.0+(3*(j-2)/(self.n-2)))
            xvals[:, j-1] = x1**aux
        return xvals

    def f1_mop_evals(self, x):
        n = x.shape[0]
        j1 = np.arange(3, self.n+1, 2)
        j2 = np.arange(2, self.n+1, 2)
        val = np.zeros(n)
        for j in j1:
            aux = 0.5*(1.0+(3*(j-2)/(self.n-2)))
            val = val + (x[:, j-1] - x[:, 0]**aux)**2
        f1 = x[:, 0] + (2.0/j1.shape[0])*val
        val = np.zeros(n)
        for j in j2:
            aux = 0.5*(1.0+(3*(j-2)/(self.n-2)))
            val = val + (x[:, j-1] - x[:, 0]**aux)**2
        f2 = 1 - np.sqrt(x[:, 0]) + (2.0/j2.shape[0])*val
        return f1, f2

    def f2_mop(self, x1):
        xvals = np.zeros((x1.shape[0], self.n))
        xvals[:, 0] = x1
        for j in range(2, self.n+1):
            xvals[:, j-1] = np.sin(6*np.pi*x1+j*np.pi/self.n)
        return xvals

    def f2_mop_evals(self, x):

        n = x.shape[0]
        j1 = np.arange(3, self.n+1, 2)
        j2 = np.arange(2, self.n+1, 2)

        val = np.zeros(n)
        for j in j1:
            val = val + (x[:, j-1]-np.sin(6*np.pi*x[:, 0]+j*np.pi/self.n))**2
        f1 = x[:, 0] + (2.0/j1.shape[0])*val

        val = np.zeros(n)
        for j in j2:
            val = val + (x[:, j-1]-np.sin(6*np.pi*x[:, 0]+j*np.pi/self.n))**2
        f2 = 1 - np.sqrt(x[:, 0]) + (2.0/j2.shape[0])*val
        return f1, f2

    def f3_mop(self, x1):
        xvals = np.zeros((x1.shape[0], self.n))
        xvals[:, 0] = x1
        j1 = np.arange(3, self.n+1, 2)
        j2 = np.arange(2, self.n+1, 2)
        for j in j1:
            xvals[:, j-1] = 0.8*x1*np.cos(6*np.pi*x1+j*np.pi/self.n)

        for j in j2:
            xvals[:, j-1] = 0.8*x1*np.sin(6*np.pi*x1+j*np.pi/self.n)
        return xvals

    def f3_mop_evals(self, x):
        n = x.shape[0]
        j1 = np.arange(3, self.n+1, 2)
        j2 = np.arange(2, self.n+1, 2)

        val = np.zeros(n)
        for j in j1:
            val = val + (x[:, j-1] - 0.8*x[:, 0]*np.cos(6*np.pi*x[:, 0]+j*np.pi/self.n))**2
        f1 = x[:, 0] + (2.0/j1.shape[0])*val

        val = np.zeros(n)
        for j in j2:
            val = val + (x[:, j-1] - 0.8*x[:, 0]*np.sin(6*np.pi*x[:, 0]+j*np.pi/self.n))**2
        f2 = 1 - np.sqrt(x[:, 0]) + (2.0/j2.shape[0])*val
        return f1, f2

    def f4_mop(self, x1):
        xvals = np.zeros((x1.shape[0], self.n))
        xvals[:, 0] = x1
        j1 = np.arange(3, self.n+1, 2)
        j2 = np.arange(2, self.n+1, 2)
        for j in j1:
            xvals[:, j-1] = 0.8*x1*np.cos((6*np.pi*x1+j*np.pi/self.n)/3)

        for j in j2:
            xvals[:, j-1] = 0.8*x1*np.sin(6*np.pi*x1+j*np.pi/self.n)
        return xvals

    def f4_mop_evals(self, x):
        n = x.shape[0]
        j1 = np.arange(3, self.n+1, 2)
        j2 = np.arange(2, self.n+1, 2)
        val = np.zeros(n)
        for j in j1:
            val = val + (x[:, j-1] - 0.8*x[:, 0]*np.cos((6*np.pi*x[:, 0]+j*np.pi/self.n)/3))**2
        f1 = x[:, 0] + (2.0/j1.shape[0])*val
        val = np.zeros(n)
        for j in j2:
            val = val + (x[:, j-1] - 0.8*x[:, 0]*np.sin(6*np.pi*x[:, 0]+j*np.pi/self.n))**2
        f2 = 1 - np.sqrt(x[:, 0]) + (2.0/j2.shape[0])*val
        return f1, f2

    def f5_mop(self, x1):
        xvals = np.zeros((x1.shape[0], self.n))
        xvals[:, 0] = x1
        j1 = np.arange(3, self.n+1, 2)
        j2 = np.arange(2, self.n+1, 2)

        for j in j1:
            xvals[:, j-1] = (0.3*(x1**2)*np.cos(24*np.pi*x1+4*j*np.pi/self.n)+0.6*x1)*np.cos(6*np.pi*x1+j*np.pi/self.n)

        for j in j2:
            xvals[:, j-1] = (0.3*(x1**2)*np.cos(24*np.pi*x1+4*j*np.pi/self.n)+0.6*x1)*np.sin(6*np.pi*x1+j*np.pi/self.n)
        return xvals

    def f5_mop_evals(self, x):
        n = x.shape[0]
        j1 = np.arange(3, self.n+1, 2)
        j2 = np.arange(2, self.n+1, 2)

        val = np.zeros(n)
        for j in j1:
            val = val + (x[:, j-1] - (0.3*(x[:, 0]**2)*np.cos(24*np.pi*x[:, 0]+4*j*np.pi/self.n)+0.6*x[:, 0])*np.cos(6*np.pi*x[:, 0] + j*np.pi/self.n))**2
        f1 = x[:, 0] + (2.0/j1.shape[0])*val

        val = np.zeros(n)
        for j in j2:
            val = val + (x[:, j-1] - (0.3*(x[:, 0]**2)*np.cos(24*np.pi*x[:, 0]+4*j*np.pi/self.n)+0.6*x[:, 0])*np.sin(6*np.pi*x[:, 0] + j*np.pi/self.n))**2
        f2 = 1 - np.sqrt(x[:, 0]) + (2.0/j2.shape[0])*val
        return f1, f2

    def f6_mop(self, x1):
        xvals = np.zeros((x1.shape[0], self.n))
        xvals[:, [0, 1]] = x1
        for j in range(3, self.n+1):
            sn = np.sin(2*np.pi * xvals[:, 0] + j*np.pi/self.n)
            xvals[:, j-1] = 2 * xvals[:, 1] * sn
        return xvals

    def f6_mop_evals(self, x):
        n = x.shape[0]

        j1 = np.arange(4, self.n+1, 3)
        j2 = np.arange(5, self.n+1, 3)
        j3 = np.arange(3, self.n+1, 3)

        val = np.zeros(n)
        for j in j1:
            val = val + (x[:, j-1] - 2*x[:, 1] * np.sin(2*np.pi*x[:, 0] + j*np.pi/self.n))**2

        f1 = np.cos(0.5*x[:, 0]*np.pi)*np.cos(0.5*x[:, 1]*np.pi)+2/j1.shape[0]*val

        val = np.zeros(n)
        for j in j2:
            val = val + (x[:, j-1] - 2*x[:, 1] * np.sin(2*np.pi*x[:, 0] + j*np.pi/self.n))**2
        f2 = np.cos(0.5*x[:, 0]*np.pi)*np.sin(0.5*x[:, 1]*np.pi)+2/j2.shape[0]*val

        val = np.zeros(n)
        for j in j3:
            val = val + (x[:, j-1] - 2*x[:, 1] * np.sin(2*np.pi*x[:, 0] + j*np.pi/self.n))**2
        f3 = np.sin(0.5*x[:, 0]*np.pi) + 2/j3.shape[0]*val

        return f1, f2, f3

    def f7_mop(self, x1):
        xvals = np.zeros((x1.shape[0], self.n))
        xvals[:, 0] = x1
        for j in range(2, self.n+1):
            eva = 0.5*(1.0+(3*(j-2)/(self.n-2)))
            xvals[:, j-1] = x1**eva
        return xvals

    def f7_mop_evals(self, x):
        n = x.shape[0]
        j1 = np.arange(3, self.n+1, 2)
        j2 = np.arange(2, self.n+1, 2)

        val = np.zeros(n)
        for j in j1:
            y = x[:, j-1]-x[:, 0]**(0.5*(1.0+(3*(j-2)/(self.n-2))))
            val = val + 4*y**2 - np.cos(8*y*np.pi) + 1
        f1 = x[:, 0] + 2/j1.shape[0]*val

        val = np.zeros(n)
        for j in j2:
            y = x[:, j-1]-x[:, 0]**(0.5*(1.0+(3*(j-2)/(self.n-2))))
            val = val + 4*y**2 - np.cos(8*y*np.pi) + 1
        f2 = 1 - np.sqrt(x[:, 0]) + 2/j2.shape[0]*val

        return f1, f2

    def f8_mop(self, x1):
        xvals = np.zeros((x1.shape[0], self.n))
        xvals[:, 0] = x1
        for j in range(2, self.n+1):
            eva = 0.5*(1.0+(3*(j-2)/(self.n-2)))
            xvals[:, j-1] = x1**eva
        return xvals

    def f8_mop_evals(self, x):
        n = x.shape[0]
        j1 = np.arange(3, self.n+1, 2)
        j2 = np.arange(2, self.n+1, 2)
        summ = np.zeros(n)
        prod = np.ones(n)
        for j in j1:
            y = x[:, j-1]-x[:, 0]**(0.5*(1.0+(3*(j-2)/(self.n-2))))
            summ = summ + y**2
            prod = prod * np.cos(20*y*np.pi/np.sqrt(j)) + 2
        f1 = x[:, 0] + (2.0/j1.shape[0])*(4*summ-2*prod)
        prod = np.ones(n)
        for j in j2:
            y = x[:, j-1]-x[:, 0]**(0.5*(1.0+(3*(j-2)/(self.n-2))))
            summ = summ + y**2
            prod = prod * np.cos(20*y*np.pi/np.sqrt(j)) + 2
        f2 = 1 - np.sqrt(x[:, 0]) + (2.0/j2.shape[0])*(4*summ-2*prod)
        return f1, f2

    def f9_mop(self, x1):
        xvals = np.zeros((x1.shape[0], self.n))
        xvals[:, 0] = x1
        for j in range(2, self.n+1):
            xvals[:, j-1] = np.sin(6*np.pi*x1+j*np.pi/self.n)
        return xvals

    def f9_mop_evals(self, x):
        n = x.shape[0]
        j1 = np.arange(3, self.n+1, 2)
        j2 = np.arange(2, self.n+1, 2)

        val = np.zeros(n)
        for j in j1:
            val = val + (x[:, j-1]-np.sin(6*np.pi*x[:, 0]+j*np.pi/self.n))**2
        f1 = x[:, 0] + (2.0/j1.shape[0])*val

        val = np.zeros(n)
        for j in j2:
            val = val + (x[:, j-1]-np.sin(6*np.pi*x[:, 0]+j*np.pi/self.n))**2
        f2 = 1 - np.power(x[:, 0], 2) + (2.0/j2.shape[0])*val
        return f1, f2

    PS_Functions_Dir = {"F1": f1_mop, "F2": f2_mop, "F3": f3_mop, "F4": f4_mop, "F5": f5_mop, "F6": f6_mop,
                        "F7": f7_mop, "F8": f8_mop, "F9": f9_mop}

    MOP_Functions_Dir = {"F1": f1_mop_evals, "F2": f2_mop_evals, "F3": f3_mop_evals, "F4": f4_mop_evals,
                         "F5": f5_mop_evals, "F6": f6_mop_evals, "F7": f7_mop_evals, "F8": f8_mop_evals, "F9": f9_mop_evals}

    def evaluate_mop_function(self, x):
        x[:, 1:] = x[:, 1:] * (self.DV_range[1, 1]-self.DV_range[1, 0]) + self.DV_range[1, 0]
        return self.MOP_Function(self, x)

    def generate_ps_samples(self, k):
        if self.n_obj == 2:
            all_ps = np.random.uniform(low=self.DV_range[0, 0], high=self.DV_range[0, 1], size=k)
            vals = self.PS_Function(self, all_ps)
            vals[:, 1:] = (vals[:, 1:]-self.DV_range[1, 0])/(self.DV_range[1, 1]-self.DV_range[1, 0])
        else:
            all_ps = np.zeros((k, 2))
            all_ps[:, 0] = np.random.uniform(low=self.DV_range[0, 0], high=self.DV_range[0, 1], size=k)
            all_ps[:, 1] = np.random.uniform(low=self.DV_range[1, 0], high=self.DV_range[1, 1], size=k)
            vals = self.PS_Function(self, all_ps)
            vals[:, 2:] = (vals[:, 2:]-self.DV_range[2, 0])/(self.DV_range[2, 1]-self.DV_range[2, 0])
        return vals
