# BOX QP solver. Implemented using numpy it used Projected-Newton Method
# as described  in "Control limited DDP"- Yuval Tassa et. al.

import theano.tensor as T
import numpy as np


class boxQP():
    def __init__(self, nx, maxiter =100, th_acceptstep = 0.1
                 ,th_grad =1e-9, reg = 1e-9):
        self.nx = nx
        self.maxiter = maxiter
        self.th_acceptstep = th_acceptstep
        self.th_grad = th_grad
        self.reg = reg
        self._x= np.zeros((nx), dtype=float)
        self.x_= np.zeros((nx), dtype=float)
        self.alphas = 1/ 2**(np.arange(10))

    def solve(self, H, q, lb, ub, xinit):
        xnew = np.zeros(self.nx)
        #warm starting
        for i in range(self.nx):
            self.x_[i] = max(min(xinit[i], ub[i]), lb[i])
        #start numerical iteration
        for k in range(self.maxiter):
            clamped_idx = []
            free_idx = []
            #compute gradient
            g_ = q + H @ self.x_
            #seperate free and clamped indices
            for j in range(self.nx):
                gj = g_[j]
                xj = self.x_[j]
                lbj = lb[j]
                ubj = ub[j]
                if (xj ==lbj and gj > 0 ) or (xj ==ubj and gj < 0 ):
                    clamped_idx.append(j)
                else:
                    free_idx.append(j)
            #check convergence
            nf = len(free_idx)
            nc = len(clamped_idx)
            if np.linalg.norm(g_, np.inf) <= self.th_grad or nf == 0:
                if not k:
                    Hff_ = np.zeros((nf, nf))
                    for p in range(nf):
                        for q in range(nf):
                            Hff_ [i,j] =  H[free_idx[i], free_idx[j]]
                    if  self.reg!=0:
                        Hff_+= np.eye(nf) * self.reg
                    Hff_inv= np.linalg.inv(Hff_)

                solution_x = self.x_
                return solution_x
            # Compute the search direction as Newton step along the free space
            qf = np.zeros((nf))
            xf = np.zeros((nf))
            xc = np.zeros((nf))
            dxf_ = np.zeros((nf))
            Hff_ = np.zeros((nf, nf))
            Hfc_ = np.zeros((nf, nf))
            for i in range(nf):
                fi= free_idx[i]
                qf[i] = q[i]
                xf[i] = self.x_[i]
                for j in range(nf):
                    Hff_[i,j] = H[fi, free_idx[j]]
                for j in range(nc):
                    xc[j] = self.x_[clamped_idx[j]]
                    Hfc_[i, j] = H[fi, clamped_idx[j]]
            if self.reg!=0:
                Hff_+= np.eye(nf)*self.reg

            Hff_inv = np.linalg.inv(Hff_)

            dxf_ = -qf
            if nc!=0:
                dxf_ -= Hfc_ @ xc
            dxf_ = Hff_inv @ dxf_
            dxf_ -= xf
            dx_= np.zeros((self.nx))
            for i in range (nf):
                dx_[free_idx[i]] =dxf_[i]
            #try different step length

            fold_= .5 * self.x_.transpose() @ (H @ self.x_) + q.transpose() @ (self.x_)
            for t in self.alphas:
                for r in range(self.nx):
                    xnew[r]= max(min((self.x_[r] + t * dx_[r]),ub[r]),lb[r])

                fnew_= .5 * xnew.transpose() @ (H @ xnew) + q. transpose() @ xnew
                if (fold_ - fnew_ > self.th_acceptstep * g_.transpose() @ (self.x_ - xnew)):
                    self.x_ = xnew;
                    break

        solution_x = self.x_
        return solution_x, fnew_


if __name__=='__main__':
    solver = boxQP(3)
    # H = 2 * np.asarray([[0.4, 0],[0, 1]], dtype=float)
    # q = np.asarray([-5, -6], dtype=float)
    # lb = np.asarray([0, 0], dtype=float)
    # ub = np.asarray([10, 10], dtype=float)
    # xinit =np.asarray([50,5], dtype=float)
    H = 2 * np.asarray([[4, 1, 2], [1, 8, 5], [2, 5, 4]], dtype=float)
    q = np.asarray([2, 3, 4], dtype=float)
    lb = np.asarray([-10, -10, -10], dtype=float)
    ub = np.asarray([10, 10, 10], dtype=float)
    xinit =np.asarray([5,5,5], dtype=float)
    D= solver.solve(H, q, lb, ub, xinit)
    print(D)

