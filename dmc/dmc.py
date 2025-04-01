import numpy as np
from scipy.stats import truncnorm


class DMC:
    def __init__(
        self,
        prior_means: np.ndarray,
        prior_sds: np.ndarray,
        param_names: tuple[str] = ('A', 'tau', 'mu_c', 'mu_r', 'b'),
        param_lower_bound: float | None = 0,
        num_obs: float | None = 200,
        tmax: int = 1200,
        dt: float = 1,
        sigma: float = 4.0,
        X0_beta_shape_fixed: int = 3,
        a_value: int = 2,
        num_conditions: int = 2
    ):
        """
        Initialize the DMC simulator ina  BayesFlow-friendly format.

        Parameters
        ----------
        prior_means : np.ndarray
            Array of prior means for the model parameters.
        prior_sds : np.ndarray
            Array of prior standard deviations for the model parameters.
        param_names : tuple of str, optional
            Names of the parameters. Default is ('A', 'tau', 'mu_c', 'mu_r', 'b').
        param_lower_bound : float or None, optional
            Lower bound for the prior.
        num_obs : float, optional
            Number of simulated trials. Default is 200.
        tmax : int, optional
            Maximum simulation time in milliseconds. Default is 1200.
        dt : float, optional
            Time step of the simulation. Default is 1.
        sigma : float, optional
            Standard deviation of the noise in the diffusion process. Default is 4.0.
        X0_beta_shape_fixed : int, optional
            Shape parameter used for beta distribution of initial states. Default is 3.
        a_value : int, optional
            Constant 'a' value used in the simulation. Default is 2.
        num_conditions : int, optional
            The number of conditions in the experiment. Default is 2.
        """

        self.num_obs = num_obs
        self.tmax = tmax
        self.dt = dt
        self.sigma = sigma
        self.param_names = param_names
        self.param_lower_bound = param_lower_bound
        self.prior_means = prior_means
        self.prior_sds = prior_sds
        self.X0_beta_shape_fixed = X0_beta_shape_fixed
        self.a_value = a_value
        self.num_conditions = num_conditions

        if num_conditions != 2:
            raise ValueError("Number of conditions must be 2 for this experiment.")

    def prior(self):
        """
        Sample from a (possibly truncated) normal distribution using prior parameters.
    
        Returns
        -------
        out : np.ndarray
            Sampled values from the (possibly truncated) normal distribution.
        """

        if self.param_lower_bound is not None:
            a = (self.param_lower_bound - self.prior_means) / self.prior_sds
            b = (np.inf - self.prior_means) / self.prior_sds
            p = truncnorm.rvs(a, b, loc=self.prior_means, scale=self.prior_sds)
            return dict(A=p[0], tau=p[1], mu_c=p[2], t0=p[3], b=p[4])
        return np.random.normal(self.prior_means, self.prior_sds)

    def trial(self, A: float, tau: float, mu_c: float, t0: float, b: float, t: np.ndarray, noise: np.ndarray):
        """
        Simulate multiple DMC trials in parallel.

        Parameters
        ----------
        A : float
            Amplitude of the control signal.
        tau : float
            Time constant for exponential decay.
        mu_c : float
            Constant drift component.
        t0 : float
            Non-decision time (in ms).
        b : float
            Decision boundary.
        t : np.ndarray
            Time array (in ms), shape (T,)
        noise : np.ndarray
            Noise samples, shape (n_trials, T)

        Returns
        -------
        rts : np.ndarray
            Response times in seconds, shape (n_trials,). -1 if no response.
        resps : np.ndarray
            Responses (1 = correct, 0 = error, -1 = no response), shape (n_trials,)
        """

        num_trials, _ = noise.shape
        dt = self.dt
        sqrt_dt_sigma = self.sigma * np.sqrt(dt)

        # Initial positions X0 for all trials
        X0 = np.random.beta(self.X0_beta_shape_fixed, self.X0_beta_shape_fixed, size=num_trials) * (2 * b) - b

        # Drift term mu(t), shape (T,)
        t_div_tau = t / tau
        exponent_term = np.exp(-t_div_tau)
        power_term = (np.exp(1) * t_div_tau / (self.a_value - 1)) ** (self.a_value - 1)
        deriv_term = ((self.a_value - 1) / t) - (1 / tau)
        mu_t = A * exponent_term * power_term * deriv_term + mu_c  # shape (T,)

        # Full drift for all trials: broadcast mu_t to (n_trials, T)
        dX = mu_t[None, :] * dt + sqrt_dt_sigma * noise  # shape (n_trials, T)
        X_shift = np.cumsum(dX, axis=1) + X0[:, None]    # shape (n_trials, T)

        # Check boundary crossings
        crossed_upper = X_shift >= b
        crossed_lower = X_shift <= -b
        crossed_any = crossed_upper | crossed_lower

        # First crossing index for each trial
        first_crossing = np.argmax(crossed_any, axis=1)
        has_crossed = np.any(crossed_any, axis=1)

        # Prepare output
        rts = np.full(num_trials, -1.0)
        resps = np.full(num_trials, -1)

        # Fill only for trials that crossed
        idx = np.where(has_crossed)[0]
        crossing_times = t[first_crossing[idx]]
        rts[idx] = (crossing_times + t0) / 1000  # convert to seconds

        # Determine response type
        resp_hit = X_shift[idx, first_crossing[idx]]
        resps[idx] = (resp_hit >= b).astype(int)

        return np.c_[rts, resps]
    
    def experiment(
        self, 
        A: float, 
        tau: float, 
        mu_c: float, 
        t0: float, 
        b: float, 
        min_num_obs: int = 50, 
        max_num_obs: int = 500
    ):
        """
        Simulate multiple DMC trials in parallel.

        Parameters
        ----------
        A : float
            Amplitude of the control signal.
        tau : float
            Time constant for exponential decay.
        mu_c : float
            Constant drift component.
        t0 : float
            Non-decision time (in ms).
        b : float
            Decision boundary.
        min_num_obs : int
            The minimum number of observations if num_obs not available.
        max_num_obs : int
            The maxmimum number of observations if num_obs not available.

        Returns
        -------
        sims : dict
            A dictionary with three keys: data - contains a (num_obs, 2) array of
            response times and choices; conditions - contains a (num_obs, ) array
            indicating the conditions; num_obs - an int indicating the number of trials.
        """
        
        num_obs = self.num_obs or np.random.randint(min_num_obs, max_num_obs+1)
        
        obs_per_condition = int(np.ceil(num_obs / self.num_conditions))
        conditions = np.repeat(np.arange(self.num_conditions), obs_per_condition)
    
        t = np.linspace(start=self.dt, stop=self.tmax, num=int(self.tmax / self.dt))

        noise = np.random.normal(size=(num_obs, self.tmax))
        
        data = np.zeros((num_obs, 2))
        
        data[:obs_per_condition] = self.trial(
            A=A, tau=tau, mu_c=mu_c, t0=t0, b=b, t=t, noise=noise[:obs_per_condition]
        )
        data[obs_per_condition:] = self.trial(
            A=-A, tau=tau, mu_c=mu_c, t0=t0, b=b, t=t, noise=noise[obs_per_condition:]
        )

        return dict(data=data, conditions=conditions, num_obs=num_obs)
