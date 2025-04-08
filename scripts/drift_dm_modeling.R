library(tidyverse)
library(latex2exp)
library(ggpubr)
library(bayestestR)
library(dRiftDM)

rm(list = ls(all.names = TRUE)) #will clear all objects includes hidden objects.
gc() #free up memrory and report the memory usage.

source('functions/R_functions.R')

### read simulated data
list_files <- lst_files('data/drift_dm/bf_simulations', '.csv')

list_files <- list_files[1:100]

data_sim <- map(list_files, read_csv) %>% 
  bind_rows() 

colnames(data_sim)<- c('num', 'RT', 'acc', 'Cond', 'ID')

data_sim <- data_sim %>% 
  mutate(Cond = ifelse(Cond == 0, 'comp', 'incomp')) %>% 
  mutate(Error = 1-acc) %>% 
  select(ID, RT, Error, Cond)

data_sim %>% 
  count(ID,Cond)

sim_ids <- data_sim$ID %>% unique()

delta_fun <- calc_stats(data_sim, 'delta_funs', minuends = 'incomp', subtrahends = 'comp')

plot(delta_fun)

dmc_model <- dmc_dm(t_max = 1.6, dt = 0.001, dx = .005)

estimate_model_ids(
  drift_dm_obj = dmc_model,
  obs_data_ids = data_sim,
  lower = c(muc = 1, b = .1, non_dec = .1, sd_non_dec = .005, tau = .01, A = .01, alpha = 2),
  upper = c(muc = 8, b = 1.2, non_dec = .7, sd_non_dec = .050, tau = .3, A = .4, alpha = 9),
  fit_procedure_name = "flanker_test_run", # a label to identify the fits
  fit_path = 'data/drift_dm/drift_dm_estimates', # to save fits in the working directory use getwd()
  use_de_optim = TRUE, # overrule the default Differential Evolution setting # TRUE for differential evolution
  # use_nmkb = TRUE, # TRUE for Nelder Mead
  # force_refit = TRUE
  )

data_fits <- load_fits_ids(path = 'data/drift_dm/drift_dm_estimates', fit_procedure_name = 'flanker_test_run', check_data = FALSE)

summary(data_fits)

params <- coef(data_fits)

hist(params)

# Input documentation:
# named_values: a named numeric vector
# sigma_old, sigma_new: the previous and target diffusion constants
# t_from_to: scaling of time (options: ms->s, s->ms, or none)
convert_prms <- function(named_values,
                         sigma_old = 4,
                         sigma_new = 1,
                         t_from_to = "ms->s") {
  # Some rough input checks
  stopifnot(is.numeric(named_values), is.character(names(named_values)))
  stopifnot(is.numeric(sigma_old), is.numeric(sigma_new))
  t_from_to <- match.arg(t_from_to, choices = c("ms->s", "s->ms", "none"))
  
  # Internal conversion function (takes a name and value pair, and transforms it)
  internal <- function(name, value) {
    name <- match.arg(
      name,
      choices = c("muc", "b", "non_dec", "sd_non_dec", "tau", "a", "A", "alpha")
    )
    
    # 1. scale for the diffusion constant
    if (name %in% c("muc", "b", "A")) {
      value <- value * (sigma_new / sigma_old)
    }
    
    # 2. scale for the time
    # determine the scaling per parameter (assuming s->ms)
    scale <- 1
    if (name %in% c("non_dec", "sd_non_dec", "tau")) scale <- 1000
    if (name %in% c("b", "A")) scale <- sqrt(1000)
    if (name %in% c("muc")) scale <- sqrt(1000) / 1000
    
    # adapt, depending on the t_from_to argument
    if (t_from_to == "ms->s") scale <- 1 / scale
    if (t_from_to == "none") scale <- 1
    
    value <- value * scale
  }
  
  # Apply the internal function to each element
  converted_values <- mapply(FUN = internal, names(named_values), named_values)
  
  return(converted_values)
}

n_sims <- nrow(coef(data_fits))

estimate_list <- list()

c(muc = 1, b = .3, non_dec = .1, sd_non_dec = .005, tau = .03, A = .01, alpha = 2)

convert_prms(
  named_values = c(muc = 1, b = .1, non_dec = .1, sd_non_dec = .005, tau = .01, A = .01, alpha = 2) , # the previous parameters
  sigma_old = 1, # diffusion constants
  sigma_new = 4,
  t_from_to = "s->ms" # how shall the time be scaled?
)


convert_prms(
  named_values = c(muc = 8, b = 1.2, non_dec = .7, sd_non_dec = .050, tau = .3, A = .4, alpha = 9) , # the previous parameters
  sigma_old = 1, # diffusion constants
  sigma_new = 4,
  t_from_to = "s->ms" # how shall the time be scaled?
)

for(i in 1:n_sims){
  
values <-  coef(data_fits)[i,2:8] %>% as.numeric()

names <- names( coef(data_fits)[i,2:8])

names(values) <- names

estimate_list[[i]] <- convert_prms(
  named_values = values , # the previous parameters
  sigma_old = 1, # diffusion constants
  sigma_new = 4,
  t_from_to = "s->ms" # how shall the time be scaled?
)

# estimate_list[[i]]['id'] <- coef(data_fits)[i,1]

}

drdm_estimates <- estimate_list %>% 
  bind_rows() %>% 
  mutate(ID =  coef(data_fits)[,1])

# save(drdm_estimates, file = 'data/drift_dm/bf_simulations_ground_truth')

####### LOAD GROUND TRUTH ##########

list_files <- lst_files('data/drift_dm/drift_dm_estimates/drdm_estimates.RData')

# list_files <- list_files[1:20]

# data_gt <- map(list_files, read_csv, show_col_types = FALSE) %>% 
#   bind_rows() 
# 
# save(data_gt, file = 'data/drift_dm/bf_simulations_ground_truth/bf_simulations_ground_truth_alldata.RData')

load('data/drift_dm/bf_simulations_ground_truth/bf_simulations_ground_truth_alldata.RData')

colnames(data_gt)<- c('num', 'A_gt', 'tau_gt', 'muc_gt', 'non_dec_gt', 'b_gt', 'ID')

recovery_data <- data_gt %>% 
  left_join(drdm_estimates, by = join_by(ID))


recovery_data_long <- recovery_data %>% 
  select(num, ID, everything()) %>% 
  pivot_longer(A_gt:alpha) %>% 
  mutate(gt_rec = case_when(
    str_detect(name, '_gt') ~ 'groundtruth',
    .default = 'estimate'
  )) %>% 
  mutate(parameter = str_replace(name, '_gt', '')) %>% 
  select(-name) %>% 
  pivot_wider(names_from = gt_rec) %>% 
  mutate(n_trials = case_when(
    str_detect(ID, '_1000') ~ 1000,
    str_detect(ID, '_200') ~ 200,
    .default = NA
  )) %>% 
  mutate(method = 'drift_dm')

recovery_data_long %>% 
  ggplot(aes(groundtruth, estimate, color = factor(n_trials)))+
  geom_point()+
  facet_wrap(~parameter, scales = 'free')



# recovery_data_long %>% View()

delta_fun <- calc_stats(data_fits, c('delta_funs', 'cafs', 'quantiles'), minuends = 'incomp', subtrahends = 'comp')

plot(delta_fun, mfrow = c(1,3))

##### LOAD BF ESTIMATES ########


# list_files <- lst_files('data/drift_dm/bf_post_samples', '.csv')
# 
# data_bf <- map(list_files, read_csv, show_col_types = FALSE) %>% 
#   bind_rows()

# save(data, file = c('data/drift_dm/bf_post_samples/bf_post_samples_data.RData'))
load('data/drift_dm/bf_post_samples/bf_post_samples_data.RData')

post_means <- data %>% 
  pivot_longer(A:b,
               names_to = 'parameter',
               values_to = 'bf_sample') %>% 
  mutate(parameter = str_replace_all(parameter, c('mu_c' = 'muc',
                                                  'mu_r' = 'non_dec'))) %>% 
  mutate(n_trials = case_when(
    str_detect(simulation_id, '_1000') ~ 1000,
    str_detect(simulation_id, '_200') ~ 200,
    .default = NA
  )) %>% 
  ungroup() %>% 
  group_by(condition, simulation_id, parameter, n_trials) %>% 
  summarise(estimate = mean(bf_sample)) %>% 
  rename('ID' = simulation_id)

data_gt_long <- data_gt %>% 
  pivot_longer(A_gt:b_gt) %>% 
  separate(name, into = c('parameter', 'gt'), sep = '_') %>% 
  mutate(parameter = str_replace(parameter, 'non', 'non_dec')) %>% 
  select(-gt) %>% 
  rename('ground_truth_value' = value)

# recovery_data_bf <- data_gt_long %>% 
#   left_join(post_means, by = join_by(ID, parameter, n_trials))

recovery_data_long_bf <- data_gt_long %>% 
  left_join(post_means, by = join_by(ID, parameter)) %>% 
  mutate(method = 'BayesFlow') %>% 
  select(-num)

recovery_data_long_drdm <- data_gt_long %>% 
  left_join(recovery_data_long, by = join_by(ID, parameter)) %>% 
  select(-num.y, -num.x, -groundtruth)


data_complete <- recovery_data_long_bf %>% 
  bind_rows(recovery_data_long_drdm)


data_complete %>% 
  ggplot(aes(ground_truth_value, estimate, color = method))+
  geom_point()+
  stat_cor()+
  geom_smooth(method = 'lm')+
  facet_wrap(factor(n_trials)~parameter, scales = 'free', ncol = 5)
