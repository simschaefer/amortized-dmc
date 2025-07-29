library(dplyr)
library(tidyr)
library(dRiftDM)
library(readr)
library(stringr)
library(tictoc)
library(here)

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

get_script_path <- function() {
  # 1. If running via Rscript
  args <- commandArgs(trailingOnly = FALSE)
  file_flag <- "--file="
  script_path <- NULL
  
  match <- grep(file_flag, args)
  if (length(match) > 0) {
    script_path <- sub(file_flag, "", args[match])
    return(normalizePath(script_path))
  }
  
  # 2. If running inside RStudio
  if (requireNamespace("rstudioapi", quietly = TRUE)) {
    if (rstudioapi::isAvailable()) {
      script_path <- rstudioapi::getSourceEditorContext()$path
      if (nzchar(script_path)) return(normalizePath(script_path))
    }
  }
  
  # 3. If sourced from R console
  # (only works if `ofile` is set by source())
  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(normalizePath(sys.frames()[[1]]$ofile))
  }
  
  # 4. Otherwise fallback to current working directory
  return(normalizePath(getwd()))
}

parent_dir <- dirname(dirname(get_script_path()))

setwd(parent_dir)

# Before running this script, simulate data using scripts/simulate_data.py!
# The network_name has to correspond with the name of the simulated data

# specify network name and number of trials as:

data_name <- 'initial_priors_sdr_estimated_500_trials_data'


# load simulated data
sim_data <- read_csv(str_c('data_complete/simulated_data/', data_name, '.csv'))

## format data:

sim_data_formatted <- sim_data %>% 
  #filter(sdr == 'estimated') %>% 
  rename('ID' = sim_idx,
         'RT' = rt,
         'Cond' = conditions) %>% 
  mutate(Error = 1-accuracy) %>% 
  mutate(Cond = ifelse(Cond == 0, 'comp', 'incomp')) %>% 
  select(ID, RT, Cond, Error) 

# convert params from sigma 4 -> sigma 1

priors <- sim_data %>%
  select(A:b) %>%
  unique()

# margin for parameter ranges:
factor <- 0.05

# min params:
min_prms <- convert_prms(named_values=c(muc = min(priors$mu_c) - min(priors$mu_c)*factor,
                                        b = min(priors$b) - min(priors$b)*factor,
                                        non_dec = min(priors$mu_r) - min(priors$mu_r)*factor,
					tau = min(priors$tau) - min(priors$tau)*factor,
                                        A = min(priors$A) - min(priors$A)*factor),
                         sigma_old = 4,
                         sigma_new = 1,
                         t_from_to = "ms->s")

min_prms["sd_non_dec"] <- .010

print(min_prms)

factor <- .01

# max params
max_prms <- convert_prms(named_values=c(muc = max(priors$mu_c) + max(priors$mu_c)*factor,
                                        b = max(priors$b) + max(priors$b) * factor,
                                        non_dec = max(priors$mu_r) + max(priors$mu_r) * factor,
                                        tau = max(priors$tau) + max(priors$tau)*factor, 
                                        A = max(priors$A) + max(priors$A)*factor),
                         sigma_old = 4,
                         sigma_new = 1,
                         t_from_to = "ms->s")

max_prms["sd_non_dec"] <- .07

print(max_prms)

# Exclude non-convergent trials
sim_data_formatted <- sim_data_formatted %>% 
  as.data.frame() %>% 
  filter(RT != -1) %>% 
  filter(Error != -1)

# Define discretization parameters:

dt <- .002

dmc_model <- dmc_dm(sigma = 1, t_max = 2.0, dt = dt, dx = .002)

# Fix starting point shape to 3:

coef(dmc_model)["alpha"] = 3

dmc_model <- modify_flex_prms(dmc_model, instr = "alpha <!>")


# Create folder for mode fit files:

name <- str_c('estimates_', data_name)

path <- str_c('data_complete/driftdm_fit/model_fits/',name)

if (!dir.exists(path)) {
  dir.create(path)
}

# How many cores are available?
parallel::detectCores()

# Check variable names of simulated data before fitting the model:
head(sim_data_formatted)

print('Start Parameter Estimation')

tic()
estimate_model_ids(
  drift_dm_obj = dmc_model,
  obs_data_ids = sim_data_formatted,
  lower = min_prms,
  upper = max_prms,
  fit_procedure_name = "flanker_data_estimated", # a label to identify the fits
  fit_path = path, # to save fits in the working directory use getwd()
  use_de_optim = TRUE, # overrule the default Differential Evolution setting # TRUE for differential evolution
  use_nmkb = FALSE, # TRUE for Nelder Mead
  force_refit = TRUE,
  de_n_cores = 64,
  progress = 1
)
end_time <- toc()

# save fittin time
fitting_time <- end_time$toc - end_time$tic

# load fits
data_fits <- load_fits_ids(path = path, fit_procedure_name = "flanker_data_estimated", check_data = TRUE)

print('Estimation Completed')

# Convert fits to tibble
params <- tibble(coef(data_fits)) %>% 
  mutate(fitting_time_10ids = fitting_time)

# create folder for fit data frames
esimates_path <- 'data_complete/driftdm_estimates'

if (!dir.exists(esimates_path)) {
    dir.create(esimates_path)}

# save data frames
write_csv(params, str_c(esimates_path, '/driftdm_estimates_',name,  '.csv'))


