import numpy as np

from bayesflow.simulators.benchmark_simulators.benchmark_simulator import BenchmarkSimulator


class DMC(BenchmarkSimulator):
    def __init__(
        self,
        prior_means,
        prior_sds,
        param_names = ['A', 'tau', 'mu_c', 'mu_r', 'b'],
        prior_restriction = 'positive_only',
        N: float = 200,
        tmax: int = 1200,
        dt: float = 1,
        sigma: float = 4.0,
        X0_beta_shape_fixed = 3,
        a_value = 2,
        max_nonconvergent = 1000
    ):
        """DMC simulated benchmark

        NOTE: the simulator scales outputs between 0 and 1.

        Parameters
        ----------
        N: float, optional, default: 200
            Number of simulated trials.
        """

        self.N = N
        self.tmax = tmax
        self.dt = dt
        self.sigma = sigma
        self.param_names = param_names
        self.prior_restriction = prior_restriction
        self.prior_means = prior_means
        self.prior_sds = prior_sds
        self.X0_beta_shape_fixed = X0_beta_shape_fixed
        self.a_value = a_value
        self.max_nonconvergent = max_nonconvergent

    def normal_lim(self,
                   lower = 0):

        if self.prior_restriction == 'unrestricted':
            out = np.random.normal(self.prior_means, self.prior_sds)

        elif self.prior_restriction == 'positive_only':
            out = np.random.normal(self.prior_means, self.prior_sds)

            # Keep redrawing where the condition is not met
            mask = out < lower
            while np.any(mask):
                # Redraw only for those values that did not meet the condition
                out[mask] = np.random.normal(self.prior_means[mask], self.prior_sds[mask])
                mask = out < lower

                ## todo: scipy.stats.truncnorm
        
        return out

    def prior(self):
        
        """Generates a random draw from a 5-dimensional (independent) prior restricted normal

        Returns
        -------
        params : np.ndarray of shape (2, )
            A single draw from the 2-dimensional prior.
        """

        normal_draws = self.normal_lim()

        return normal_draws

    def trial(self, params, t, noise):

        A = params[0]
        tau = params[1]
        mu_c = params[2]
        t0 = params[3]
        b = params[4]

        X0 = np.random.beta(self.X0_beta_shape_fixed, self.X0_beta_shape_fixed)*(2*b)-b

        mu = A * np.exp((-t / tau)) * (np.exp(1) * t / (self.a_value - 1) / tau)**(self.a_value - 1) * ((self.a_value - 1) / t - 1 / tau) + mu_c
        dX = mu * self.dt + self.sigma * np.sqrt(self.dt) * noise
        X_shift = np.cumsum(dX) + X0


        if np.any(X_shift >= b) or np.any(X_shift <= -b):
            d = min(t[(X_shift >= b) | (X_shift <= -b)])

            rt = (d + t0)/1000

            boundary_hit = X_shift[np.where(t == d)][0]

            if boundary_hit >= b:
                # correct response
                resp = 1
            else:
                # wrong response
                resp = 0

        else:
            rt = resp = -1
        return rt, resp

    def experiment(self, params, nonconvergent_warning = True):

        # n_obs = batchable_context.shape[0]
        n_conditions = 2
        obs_per_condition = int(np.ceil(self.N / n_conditions))
        condition = np.arange(n_conditions)
        condition = np.repeat(condition, obs_per_condition)
        np.random.shuffle(condition)

        out = np.zeros((int(self.N), 2))  # Ensure N is int
        context = condition[:int(self.N)].copy() 

        t = np.linspace(start=self.dt, stop=self.tmax, num=int(self.tmax / self.dt))

        noise = np.random.randn(len(t))

        for n in range(0, int(self.N)):

            params_trial = params.copy()

            # adjust A (congruent vs. incongruent trials)
            params_trial[0] = params_trial[0] if context[n] == 0 else -params_trial[0]

            rt = -1
            counter = 0
            while rt < 0:

                counter += 1

                rt, resp = self.trial(params_trial, t = t, noise = noise)

                if counter > self.max_nonconvergent:
                    if nonconvergent_warning:
                        print(f'WARNING: RESAMPLING DIFFUSION PROCESS DID NOT DIVERGE AFTER {self.max_nonconvergent} REPETITIONS!')

                    resp = -1

                    break
            out[n, 0] = rt
            out[n, 1] = resp
            
        return dict(rt = out[:, 0], acc = out[:, 1], context = context, N = self.N)