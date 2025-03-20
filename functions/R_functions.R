require(tidyverse)


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

filter_list <- function(list_files, selected_conditions){
  
  selected_condition <- paste(selected_conditions, collapse = "|")
  
  # Filter file paths containing any of these numbers
  filtered_files <- list_files[str_detect(list_files, selected_condition)]
  
  return(filtered_files)
  
} 



add_labels <- function(data, variables = c('parameter', 'x0_var')) {
  
  if('parameter' %in% variables){
  data <- data %>% 
    mutate(parameter_label = str_replace_all(parameter, c('mu_r' = 'mu \\\\ _r',
                                                    'mu_c' = 'mu \\\\ _c')))}
  if('x0_var' %in% variables){
  data <- data %>% 
    mutate(x0_var_label = str_replace_all(x0_var, c('fixed' = 'X_0 = 0',
                                                    'trial' = 'X_0  \\\\sim  beta \\\\ (3,3)')))}
  
  return(data)

}

model_title_to_condition <- function(data){
  
  data <- data %>% 
    mutate(condition = str_extract(model_title, "(?<=condition)\\d+") %>% 
             as.numeric()) 

  return(data)
  }

prior_categories <- function(data){
  
  
  winner_conditions_simulations <-  c(496, 506, 516, 526, 536, 546, 556, 566)
  
  data <- data %>% 
    mutate(priors = case_when(
    condition >= 1145 & condition<= 1152 ~ 'Out-of-Sample',
    condition > 1152 & condition < 1161 ~ 'acdc wide',
    condition >= 1161 &condition <= 1168 ~ 'ACDC',
    condition %in% winner_conditions_simulations ~ 'Winner Simulation',
    .default = 'All Simulations'
  ))
  
  return(data)
}

read_rdata <- function(file) {
  env <- new.env()  # Load into a new environment to capture the object
  load(file, envir = env)
  as.list(env)  # Convert the environment to a list
}


lst_files <- function(path, pattern){
  
  lst_files <- str_c(path, '/', list.files(path, pattern = pattern))
  
  return(lst_files)
}

condition_specs <- function(design_grid_path = 'data/design_grid6.csv'){

# dg6 <- read_csv('data/design_grid6.csv')
dg6 <- read_csv(design_grid_path)

condition_specs <- dg6 %>% 
  # filter(condition %in% unique(data_long$condition)) %>% 
  select(condition, n_epochs, max_obs, sd_r_var, a_prior, x0_var, contains('_prior_'), dropout)

return(condition_specs)
}

add_condition_specs <- function(data, design_grid_path = 'data/design_grid6.csv'){
  
  # dg6 <- read_csv('data/design_grid6.csv')
  dg6 <- read_csv(design_grid_path)
  
  condition_specs <- dg6 %>% 
    # filter(condition %in% unique(data_long$condition)) %>% 
    select(condition, n_epochs, max_obs, sd_r_var, a_prior, x0_var, contains('_prior_'))
  
  data <- data %>% 
    left_join(condition_specs, by = join_by(condition))
  
  return(data)
}

appender <- function(string) {
  TeX(paste0("$", string, "$"))}

# yellow : '#fde725'
values = c('#440154', '#31688e','#35b779','#f9795d')
# values = c('#12436D', '#28A197', '#801650', '#F46A25')
# '#3D76A1'
# 
# '#4A85B4'
# 
# '#5B96C8'
# 
# '#29895B'
# 
# '#23734D'
# 
# '#5A1E66'
# 
# '#6FC99A'
# 
# '#89D4AE'

# '#FFB6B6'
# 
# '#FFA3A3'
# 
# '#FF9E8F'
# '#E85A5A'
# 
# '#FF8C8C'
# 
# '#FF7878'
# 
# '#31688e'
# 
# '#F06565'
# 
# '#85D485'
# 
# '#78FF78'
# 
# '#2F855A'
# 
# '#85D485'
# 
# '#26828e'
# 
# '#4A4A4A'
# 
# 
# '#1D4B4A'
# '#1C1C1C'
# '#5C2D6D'

model_complexity_color <- function(values = c('#440154', '#31688e','#6DBC6F','#FF7878')){
  list(scale_color_manual(breaks = c('gamma.estimated',
                                     'gamma.fixed',
                                     'fixed.estimated',
                                     'fixed.fixed'),
                          labels = unname(TeX(c("$\\sd_r \\, estimated, a \\, estimated$",
                                                "$\\sd_r \\, fixed, a \\, estimated$",
                                                "$\\sd_r \\, estimated, a \\,fixed$",
                                                "$\\sd_r \\, fixed, a \\, fixed$"))),
                          values = values),
       labs(colour="Model Complexity"),
       theme_minimal())
  
  
}

model_complexity_fill <- function(values = c('#440154', '#31688e','#35b779','#fde725')){

    list(scale_fill_manual(breaks = c('gamma.estimated',
                                  'gamma.fixed',
                                  'fixed.estimated',
                                  'fixed.fixed'),
                       labels = unname(TeX(c("$\\sd_r \\, estimated, a \\, estimated$",
                                             "$\\sd_r \\, fixed, a \\, estimated$",
                                             "$\\sd_r \\, estimated, a \\,fixed$",
                                             "$\\sd_r \\, fixed, a \\, fixed$"))),
                       values = values),
    labs(fill="Model Complexity"),
    theme_minimal())
  
}



filter_condition <- function(model_titles, lowest_condition_num = 335, condition_list = c()){
  model_titles <- model_titles[str_extract(model_titles, "condition\\d+", group = NULL) %>% 
                                 str_replace('condition', '') %>% 
                                 as.numeric() >= lowest_condition_num | str_extract(model_titles, "condition\\d+", group = NULL) %>% 
                                 str_replace('condition', '') %>% 
                                 as.numeric() %in% condition_list] %>% 
    na.omit() 
  
  attributes(model_titles) <- NULL
  
  return(model_titles)
}

library(stats)

get_coverage_probs <- function(z, u) {
  # Vectorized function to compute the minimal coverage probability for uniform ECDFs
  # given evaluation points z and a sample of samples u.
  
  # Parameters
  # ----------
  # z  : Numeric vector of evaluation points.
  # u  : Numeric matrix of simulated draws (samples) from U(0, 1).
  
  N <- ncol(u)
  
  # Compute empirical CDF values
  F_m <- rowSums(outer(z, u, ">=")) / N
  
  # Compute binomial cumulative distribution functions
  bin1 <- pbinom(N * F_m, N, z)
  bin2 <- pbinom(N * F_m - 1, N, z)
  
  # Compute gamma values
  gamma <- 2 * apply(pmin(bin1, 1 - bin2), 1, min)
  
  return(gamma)
}

simultaneous_ecdf_bands <- function(num_samples, 
                                    num_points = NULL, 
                                    num_simulations = 1000, 
                                    confidence = 0.95, 
                                    eps = 1e-5, 
                                    max_num_points = 1000) {
  # Computes the simultaneous ECDF bands through simulation according to
  # the algorithm described in Section 2.2.
  
  # Parameters
  # ----------
  # num_samples     : Integer, the sample size used for computing the ECDF.
  # num_points      : Integer, optional. Defaults to NULL.
  # num_simulations : Integer, optional. Defaults to 1000.
  # confidence      : Numeric in (0, 1), optional. Defaults to 0.95.
  # eps             : Numeric, optional. Defaults to 1e-5.
  # max_num_points  : Integer, optional. Defaults to 1000.
  
  # Returns
  # -------
  # A list containing `alpha`, `z`, `L`, and `U`.
  
  N <- num_samples
  
  # Determine the number of evaluation points
  if (is.null(num_points)) {
    K <- min(N, max_num_points)
  } else {
    K <- min(num_points, max_num_points)
  }
  
  M <- num_simulations
  
  # Specify evaluation points
  z <- seq(0 + eps, 1 - eps, length.out = K)
  
  # Simulate M samples of size N
  u <- matrix(runif(M * N), nrow = M, ncol = N)
  
  # Get alpha
  alpha <- 1 - confidence
  
  # Compute minimal coverage probabilities
  gammas <- get_coverage_probs(z, u)
  
  # Use insights from paper to compute lower and upper confidence interval
  gamma <- quantile(gammas, probs = alpha)
  
  L <- qbinom(gamma / 2, N, z) / N
  U <- qbinom(1 - gamma / 2, N, z) / N
  
  return(list(alpha = alpha, z = z, L = L, U = U))
}

slurm_id_ascharacter <- function(data){
  
  data <- data %>% 
    mutate(slurm_id = as.character(slurm_id))
}

check_conditions <- function(data){
  
  conds <- data$condition %>% 
    unique() %>% 
    sort() 
  
  all_conds <- min(conds):max(conds)
  
  return(all_conds[!all_conds %in% conds])
  
}