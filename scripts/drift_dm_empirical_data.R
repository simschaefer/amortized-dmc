
####### SETUP ########
library(tidyverse)
library(latex2exp)
library(ggpubr)
library(bayestestR)
library(dRiftDM)
library(broom)

rm(list = ls(all.names = TRUE)) #will clear all objects includes hidden objects.
gc() #free up memrory and report the memory usage.

source('functions/R_functions.R')

data_emp <- read_csv('data/model_data/experiment_data_spacing_downsampled.csv') %>% 
  mutate(Cond = ifelse(congruency_num == 0, 'incomp', 'comp')) %>% 
  mutate(spacing = ifelse(spacing_num == 0, 'wide', 'narrow')) %>% 
  rename('RT' = rt,
         'ID' = participant) %>% 
  mutate(Error = 1-accuracy) %>% 
  select(ID, RT, Error, Cond, spacing)

data_narrow <- data_emp %>% 
  filter(spacing == 'narrow') %>% 
  select(-spacing)

data_wide <- data_emp %>% 
  filter(spacing == 'wide')%>% 
  select(-spacing)

dmc_model <- dmc_dm(t_max = 1.3, dt = 0.001, dx = .05)

estimate_model_ids(
  drift_dm_obj = dmc_model,
  obs_data_ids = data_narrow,
  lower = c(muc = 1, b = .1, non_dec = .1, sd_non_dec = .005, tau = .01, A = .01, alpha = 2),
  upper = c(muc = 8, b = 1.2, non_dec = .7, sd_non_dec = .050, tau = .2, A = .4, alpha = 9),
  fit_procedure_name = "spacing_narrow", # a label to identify the fits
  fit_path = 'data/drift_dm/drift_dm_estimates_empirical', # to save fits in the working directory use getwd()
  use_de_optim = TRUE, # overrule the default Differential Evolution setting # TRUE for differential evolution
  use_nmkb = FALSE, # TRUE for Nelder Mead
  force_refit = TRUE
)

data_fits <- load_fits_ids(path = 'data/drift_dm/drift_dm_estimates_empirical', fit_procedure_name = 'spacing_narrow', check_data = FALSE)

summary(data_fits)

drdm_estimates <- coef(data_fits) %>% 
  as_tibble()


estimate_list <- list()

for(i in 1:length(drdm_estimates$ID %>% unique)){
  
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

###### LOAD BF DATA ######

winner_conditions <- c(496, 506, 516, 526, 536, 546, 556, 566)

list_files <- lst_files('data/posterior_experiment/posterior_spacing_individual_level', '.csv') %>% 
  str_replace('_experiment_', '_condition') %>% 
  filter_condition(1145, condition_list = winner_conditions) %>% 
  str_replace('_condition', '_experiment_') 

bf_estimates <- map(list_files, read_csv, show_col_types = FALSE) %>% 
  bind_rows()

post_means <- bf_estimates %>% 
  select(A:id, condition) %>% 
  rename('ID' = id,
         'muc' = mu_c,
         'non_dec' = mu_r) %>% 
  pivot_longer(A:b,
               names_to = 'parameter',
               values_to = 'posterior_sample') %>% 
  group_by(ID, parameter, condition) %>% 
  reframe(post_mean = mean(posterior_sample),
          post_median = median(posterior_sample),
          post_sd = sd(posterior_sample)) 

condition_spec = condition_specs()


validation_data <- drdm_estimates %>% 
  pivot_longer(muc:alpha, 
               names_to = 'parameter',
               values_to = 'drdm_estimate') %>% 
  left_join(post_means, by = join_by(ID, parameter)) %>% 
  left_join(condition_spec, by = join_by(condition)) 


validation_data %>% 
  # left_join(condition_specs, by = join_by(condition)) %>% 
  select(condition, drdm_estimate, post_median, post_mean,post_sd, a_prior, sd_r_var, x0_var, parameter) %>% 
  drop_na() %>% 
  filter(condition %in% winner_conditions) %>% 
  ggplot(aes(drdm_estimate, post_mean, color = interaction(a_prior, sd_r_var)))+
  geom_point()+
  geom_abline(aes(intercept = 0, slope = 1))+
  facet_wrap(x0_var~parameter, scales = 'free', ncol = 5)+
  geom_smooth(method = 'lm')+
  stat_cor()+
  model_complexity_color()+
  labs(title = 'Validity dRiftDM vs. BF, winner simulations only')+
  geom_errorbar(aes(ymin = post_mean - post_sd, ymax = post_mean + post_sd), alpha = 0.5)

ggsave('plots/drift_dm_validation/drift_dm_validation_empirical_data_narrow_winner_conditions.jpg', width = 10, height = 10)


validation_data %>% 
  # left_join(condition_specs, by = join_by(condition)) %>% 
  select(condition, drdm_estimate, post_median, post_mean,post_sd, a_prior, sd_r_var, x0_var, parameter) %>% 
  drop_na() %>% 
  filter(condition %in% 1145:1152) %>% 
  ggplot(aes(drdm_estimate, post_mean, color = interaction(a_prior, sd_r_var)))+
  geom_point()+
  geom_abline(aes(intercept = 0, slope = 1))+
  facet_wrap(x0_var~parameter, scales = 'free', ncol = 5)+
  geom_smooth(method = 'lm')+
  stat_cor()+
  model_complexity_color()+
  labs(title = 'Validity dRiftDM vs. BF, Out-Of-Sample only')+
  geom_errorbar(aes(ymin = post_mean - post_sd, ymax = post_mean + post_sd), alpha = 0.5)

ggsave('plots/drift_dm_validation/drift_dm_validation_empirical_data_narrow_retrain_oof.jpg', width = 10, height = 10)


validation_data %>% 
  # left_join(condition_specs, by = join_by(condition)) %>% 
  select(condition, drdm_estimate, post_median, post_mean,post_sd, a_prior, sd_r_var, x0_var, parameter) %>% 
  drop_na() %>% 
  filter(condition %in% 1161:1168) %>% 
  ggplot(aes(drdm_estimate, post_mean, color = interaction(a_prior, sd_r_var)))+
  geom_point()+
  geom_abline(aes(intercept = 0, slope = 1))+
  facet_wrap(x0_var~parameter, scales = 'free', ncol = 5)+
  geom_smooth(method = 'lm')+
  stat_cor()+
  model_complexity_color()+
  labs(title = 'Validity dRiftDM vs. BF, ACDC only')+
  geom_errorbar(aes(ymin = post_mean - post_sd, ymax = post_mean + post_sd), alpha = 0.5)

ggsave('plots/drift_dm_validation/drift_dm_validation_empirical_data_narrow_retrain_acdc.jpg', width = 10, height = 10)
