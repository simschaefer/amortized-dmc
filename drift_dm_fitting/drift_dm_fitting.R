library(dRiftDM)
library(tidyverse)

getwd()

data <- read_csv('Documents/bf_dmc/data/simulated_data/dmc_optimized_winsim_priors_sdr_estimated_200_795738_data.csv')

data_long <- data %>% 
  pivot_longer(c(A, tau, mu_c, mu_r, b, sd_r), names_to = 'parameter', values_to = 'ground_truth')

data_long$network_name %>% unique()

bf_estimates <- read_csv('Documents/bf_dmc/data/simulated_data/dmc_optimized_winsim_priors_sdr_estimated_200_795738_samples.csv')

bf_estimates_long <- bf_estimates %>% 
  pivot_longer(A:sd_r, names_to = 'parameter', values_to = 'sample') %>% 
  group_by(network_name,parameter,sim_idx, n_obs, priors, sdr) %>% 
  summarise(post_mean = mean(sample))


bf_estimates_long$network_name %>% unique()

data_long <- data_long%>% 
  select(network_name, priors, sdr, parameter, ground_truth) %>% 
  unique()

data_long %>% 
  select(ground_truth, parameter, sim_idx, n_obs,network_name, priors, sdr) %>% 
  left_join(bf_estimates_long, by = join_by(parameter, sim_idx, n_obs, network_name, sdr, priors)) 

dmc <- dmc_dm()

summary(dmc)

prms_solve(ddm)["t_max"] <- 1.5


# attach the data to the model
obs_data(ddm) <- data

# now call the estimation routine
ddm <- estimate_model(
  drift_dm_obj = ddm,
  lower = c(muc = 1, b = .3, non_dec = .1, sd_non_dec = .005, tau = .03, A = .01, alpha = 2),
  upper = c(muc = 6, b = .9, non_dec = .5, sd_non_dec = .050, tau = .12, A = .15, alpha = 9),
  use_de_optim = FALSE, # overrule the default Differential Evolution setting
  use_nmkb = TRUE
)