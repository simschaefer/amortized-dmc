library(tidyverse)
library(latex2exp)
library(ggpubr)
library(bayestestR)

rm(list = ls(all.names = TRUE)) #will clear all objects includes hidden objects.
gc() #free up memrory and report the memory usage.

source('functions/R_functions.R')

######## EMPIRICAL ########

data_emp_resim <- lst_files('data/ppc_delta_functions/ppc_delta_functions_empirical', '.csv') %>% 
  map(read_csv) %>% 
  map(slurm_id_ascharacter) %>% 
  bind_rows()


data_emp_resim_long <- data_emp_resim %>% 
  mutate(congruency = ifelse(congruency ==1, 'incomp', 'comp')) %>% 
  pivot_longer(starts_with('qu')) %>% 
  select(id, congruency, condition, spacing, name, value) %>% 
  mutate(quantile = str_replace(name, '_rt', '')) %>% 
  select(-name) %>% 
  rename('resim' = value) %>% 
  pivot_wider(names_from = congruency, values_from = resim) %>% 
  unnest(c(comp, incomp)) %>% 
  mutate(mean_qu = (comp+incomp)/2,
         delta = incomp-comp)
## empirical data

data_emp <- read_csv('data/model_data/experiment_data_wide.csv') %>% 
  bind_rows(read_csv('data/model_data/experiment_data_narrow.csv'))

data_emp %>% count(participant) %>% 
  ggplot(aes(n))+
  geom_histogram()

data_emp_long <- data_emp %>% 
  group_by(participant, congruency_num, spacing_num) %>% 
  summarise(empirical = quantile(rt, seq(from = 0.1, to = 1, by = .1)),
            quantile = paste0('qu', 1:10*10)) %>% 
  mutate(congruency = ifelse(congruency_num == 1, 'incomp', 'comp')) %>% 
  ungroup() %>% 
  select(-congruency_num) %>% 
  mutate(spacing = ifelse(spacing_num == 1, 'narrow', 'wide')) %>% 
  select(-spacing_num) %>% 
  pivot_wider(names_from = congruency, values_from = empirical) %>% 
  unnest(c(comp, incomp)) %>% 
  mutate(mean_qu = (comp+incomp)/2,
         delta = incomp-comp) %>% 
  rename('id' = participant)

data_emp_long$id %>% unique()

data_emp_long %>% 
  count(id) %>% 
  ggplot(aes(n))+
  geom_histogram()

data_emp_long %>% 
  ungroup() %>% 
  count(spacing)%>% 
  ggplot(aes(n))+
  geom_histogram()

data_emp_long %>% 
  ggplot(aes(mean_qu, delta, color = spacing))+
  geom_point()+
  geom_line(aes(group = interaction(id, spacing)))

###### MODEL FIT #####

# example of one delta function
data_emp_long %>% 
  filter(id == 8704) %>% 
  ggplot(aes(mean_qu, delta))+
  geom_point()+
  facet_wrap(~spacing)


# data_emp_resim_long %>% 
#   left_join(data_emp_long, by = join_by(spacing, id, quantile), suffix = c('_resimulated', '_empirical')) %>% 
#   ggplot(aes(delta_empirical, delta_resimulated, color = spacing))+
#   geom_point()

data_emp_resim_long %>% 
  left_join(data_emp_long, by = join_by(spacing, id, quantile), suffix = c('_resimulated', '_empirical')) %>% 
  select(condition, id, quantile, spacing,starts_with('mean_qu'), starts_with('delta') ) %>% 
  pivot_longer(mean_qu_resimulated:delta_empirical,
               names_to = 'emp_resim', values_to = 'value') %>% 
  mutate(emp_resim = str_replace(emp_resim, 'mean_qu', 'meanqu')) %>% 
  separate(emp_resim, into = c('par', 'data_source'), sep = '_') %>% 
  pivot_wider(names_from = par, values_from = value) %>% 
  ggplot(aes(meanqu, delta, color = spacing))+
  geom_point()

data_complete <- data_emp_resim_long %>% 
  left_join(data_emp_long, by = join_by(spacing, id, quantile), suffix = c('_resimulated', '_empirical')) %>% 
  select(condition, id, quantile, spacing,starts_with('mean_qu'), starts_with('delta') ) %>% 
  pivot_longer(mean_qu_resimulated:delta_empirical,
               names_to = 'emp_resim', values_to = 'value') %>% 
  mutate(emp_resim = str_replace(emp_resim, 'mean_qu', 'meanqu')) %>% 
  separate(emp_resim, into = c('par', 'data_source'), sep = '_') %>% 
  pivot_wider(names_from = par, values_from = value) %>% 
  add_condition_specs() %>% 
  prior_categories() %>% 
  mutate(priors = case_when(
    data_source == 'empirical' ~ 'Empirical',
    .default = priors
  )) %>% 
  mutate(a_prior = ifelse(data_source == 'empirical', 'empirical', a_prior),
         sd_r_var = ifelse(data_source == 'empirical', 'empirical', sd_r_var)
         # x0_var = ifelse(data_source == 'empirical', NA, x0_var)
         )


data_complete %>% 
  filter(quantile != 'qu100') %>% 
  ggplot(aes(meanqu, delta, color = data_source))+
  geom_point()+
  geom_line(aes(group = interaction(spacing, id, data_source)))+
  facet_wrap(condition~spacing)

data_complete %>% 
  filter(quantile != 'qu100') %>% 
  ggplot(aes(meanqu, delta, color = spacing))+
  geom_point(alpha = 0.5)+
  geom_line(aes(group = interaction(spacing, id, data_source)))+
  facet_wrap(~condition)

ids <- data_complete$id %>% unique()

library(glue)

idx <- 1:length(ids)

for(i in ids){
data_complete %>% 
  # drop_na() %>% 
  filter(id == i) %>% 
  filter(quantile != 'qu100') %>% 
  # filter(spacing == 'narrow') %>%
  filter(priors %in% c("Out-of-Sample", "Empirical")) %>% 
  ggplot(aes(meanqu, delta, color = interaction(a_prior, sd_r_var), shape = factor(priors), linetype =factor(priors)))+
  geom_point()+
  geom_line(aes(group = interaction(condition, spacing, id, data_source)))+
  facet_grid(x0_var~spacing)+
  labs(title = glue('Out-Of-Sample, id: {i}'))+
  scale_color_manual(breaks = c('empirical.empirical',
    'gamma.estimated',
                                     'gamma.fixed',
                                     'fixed.estimated',
                                     'fixed.fixed'),
                          labels = unname(TeX(c("$Empirical$",
                                                "$\\sd_r \\, estimated, a \\, estimated$",
                                                "$\\sd_r \\, fixed, a \\, estimated$",
                                                "$\\sd_r \\, estimated, a \\,fixed$",
                                                "$\\sd_r \\, fixed, a \\, fixed$"))),
                          values = c('#000000' ,'#440154', '#31688e','#6DBC6F','#FF7878'))+
       labs(colour="Model Complexity")+
  theme_minimal(15)

ggsave(glue('plots/experiment/plots_delta_functions/oof/plots_delta_functions_oof{i}.jpg'), width = 8, height = 8)
}


for(i in ids){
  data_complete %>% 
    # drop_na() %>% 
    filter(id == i) %>% 
    filter(quantile != 'qu100') %>% 
    # filter(spacing == 'narrow') %>%
    filter(priors %in% c("ACDC", "Empirical")) %>% 
    ggplot(aes(meanqu, delta, color = interaction(a_prior, sd_r_var), shape = factor(priors), linetype =factor(priors)))+
    geom_point()+
    geom_line(aes(group = interaction(condition, spacing, id, data_source)))+
    facet_grid(x0_var~spacing)+
    labs(title = glue('ACDC, id: {i}'))+
    scale_color_manual(breaks = c('empirical.empirical',
                                  'gamma.estimated',
                                  'gamma.fixed',
                                  'fixed.estimated',
                                  'fixed.fixed'),
                       labels = unname(TeX(c("$Empirical$",
                                             "$\\sd_r \\, estimated, a \\, estimated$",
                                             "$\\sd_r \\, fixed, a \\, estimated$",
                                             "$\\sd_r \\, estimated, a \\,fixed$",
                                             "$\\sd_r \\, fixed, a \\, fixed$"))),
                       values = c('#000000' ,'#440154', '#31688e','#6DBC6F','#FF7878'))+
    labs(colour="Model Complexity")+
    theme_minimal(15)
  
  ggsave(glue('plots/experiment/plots_delta_functions/acdc/plots_delta_functions_acdc{i}.jpg'), width = 8, height = 8)
}


for(i in ids){
data_complete %>% 
  # drop_na() %>% 
  filter(id == i) %>% 
  filter(quantile != 'qu100') %>% 
  # filter(spacing == 'narrow') %>%
  filter(priors %in% c("Winner Simulation", "Empirical")) %>% 
  ggplot(aes(meanqu, delta, color = interaction(a_prior, sd_r_var), shape = factor(priors), linetype =factor(priors)))+
  geom_point()+
  geom_line(aes(group = interaction(condition, spacing, id, data_source)))+
  facet_grid(x0_var~spacing)+
  labs(title = glue('Winner Simulation, id: {i}'))+
  scale_color_manual(breaks = c('empirical.empirical',
                                'gamma.estimated',
                                'gamma.fixed',
                                'fixed.estimated',
                                'fixed.fixed'),
                     labels = unname(TeX(c("$Empirical$",
                                           "$\\sd_r \\, estimated, a \\, estimated$",
                                           "$\\sd_r \\, fixed, a \\, estimated$",
                                           "$\\sd_r \\, estimated, a \\,fixed$",
                                           "$\\sd_r \\, fixed, a \\, fixed$"))),
                     values = c('#000000' ,'#440154', '#31688e','#6DBC6F','#FF7878'))+
  labs(colour="Model Complexity")+
  theme_minimal(15)

ggsave(glue('plots/experiment/plots_delta_functions/winner_simulations/plots_delta_functions_winner_simulations{i}.jpg'), width = 8, height = 8)

}



data_complete

###### DRIFT DM ESTIMATES #######



####### SETUP ########

data_emp <- read_csv('data/model_data/experiment_data_wide.csv') %>% 
  bind_rows(read_csv('data/model_data/experiment_data_narrow.csv'))


data_emp <- data_emp%>% 
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
  fit_path = 'data/drift_dm/drift_dm_estimates_empirical_narrow', # to save fits in the working directory use getwd()
  use_de_optim = TRUE, # overrule the default Differential Evolution setting # TRUE for differential evolution
  use_nmkb = FALSE, # TRUE for Nelder Mead
  force_refit = TRUE
)

data_fits_narrow <- load_fits_ids(path = 'data/drift_dm/drift_dm_estimates_empirical_narrow', fit_procedure_name = 'spacing_narrow', check_data = FALSE)

summary(data_fits_narrow)

drdm_estimates_narrow <- coef(data_fits_narrow) %>% 
  as_tibble()


dmc_model <- dmc_dm(t_max = 1.3, dt = 0.001, dx = .05)

estimate_model_ids(
  drift_dm_obj = dmc_model,
  obs_data_ids = data_wide,
  lower = c(muc = 1, b = .1, non_dec = .1, sd_non_dec = .005, tau = .01, A = .01, alpha = 2),
  upper = c(muc = 8, b = 1.2, non_dec = .7, sd_non_dec = .050, tau = .2, A = .4, alpha = 9),
  fit_procedure_name = "spacing_wide", # a label to identify the fits
  fit_path = 'data/drift_dm/drift_dm_estimates_empirical_wide', # to save fits in the working directory use getwd()
  use_de_optim = TRUE, # overrule the default Differential Evolution setting # TRUE for differential evolution
  use_nmkb = FALSE, # TRUE for Nelder Mead
  force_refit = TRUE
)

data_fits_wide <- load_fits_ids(path = 'data/drift_dm/drift_dm_estimates_empirical_wide', fit_procedure_name = 'spacing_wide', check_data = FALSE)

summary(data_fits_wide)

# convert_fits(data_fits_wide)
convert_fits(data_fits_narrow)

stats <- calc_stats(object = data_fits_narrow, type = c('delta_funs'), minuends = 'incomp', subtrahends = 'comp') 

plot(stats)

stats

delta_fun_drdm_narrow <- stats %>% 
  as_tibble() %>% 
  filter(Source == 'pred') %>% 
  mutate(priors = 'dRiftDM',
         quantile = str_c('qu', Prob*100),
         spacing = 'narrow') %>% 
  select(ID, priors, quantile, spacing, Delta_incomp_comp, Avg_incomp_comp) %>% 
  rename(
         'meanqu' = Avg_incomp_comp,
         'id' = ID) %>% 
  mutate(data_source = "resimulated",
         sd_r_var = 'estimated',
         a_prior = "estimated",
         a_prior = "fixed",
         delta = Delta_incomp_comp*100)

data_complete <- data_complete %>% 
  bind_rows(delta_fun_drdm_narrow)

i <- 275
data_complete %>% 
  # drop_na() %>% 
  filter(id == i) %>%
  filter(quantile != 'qu100') %>% 
  # filter(spacing == 'narrow') %>%
  filter(priors %in% c("Winner Simulation", "Empirical", "dRiftDM")) %>% 
  ggplot(aes(meanqu, delta, color = interaction(a_prior, sd_r_var), shape = factor(priors), linetype =factor(priors)))+
  geom_point()+
  geom_line(aes(group = interaction(condition, spacing, id, data_source)))+
  facet_grid(x0_var~spacing)+
  labs(title = glue('Winner Simulation, id: {i}'))+
  scale_color_manual(breaks = c('empirical.empirical',
                                'gamma.estimated',
                                'gamma.fixed',
                                'fixed.estimated',
                                'fixed.fixed'),
                     labels = unname(TeX(c("$Empirical$",
                                           "$\\sd_r \\, estimated, a \\, estimated$",
                                           "$\\sd_r \\, fixed, a \\, estimated$",
                                           "$\\sd_r \\, estimated, a \\,fixed$",
                                           "$\\sd_r \\, fixed, a \\, fixed$"))),
                     values = c('#000000' ,'#440154', '#31688e','#6DBC6F','#FF7878'))+
  labs(colour="Model Complexity")+
  theme_minimal(15)
