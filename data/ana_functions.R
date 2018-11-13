# some helper functions
library(tidyverse)
library(cowplot)
library(modelr)
library(ez)
library(rootSolve)
library(nleqslv)
library(ggsignif) 
library(ggpubr)
library(reshape2)
theme_set(theme_classic())

#' Estimate d-prime and C
#' calculate response sensitivity and bias using signal detection theory (sdt)
#' @param df experimental raw data
#' @return dprime and bias C
sdt_dprime <- function(df){
  # prepare group variables
  if ('reward' %in% colnames(df)){ # second experiment with reward
    gp_var = c("sub","target","dyn","setsize","reward")
  } else {
    gp_var = c("sub","target","dyn","setsize")
  }
  gp_vars = syms(gp_var) # prepare the group variables
  
  # subject-wise
  sdt_sub <- df %>%  filter(resp >0) %>% # filter valid trials
    group_by(!!!gp_vars) %>% 
    summarise(mresp = mean(2-resp))%>% # calculate positive response subject-wise
    spread(target, mresp ) %>% # rearrange hit (Prsent) and FA (Absent) in one row
    mutate(dprime = qnorm(pmin(Present,0.999))-qnorm(pmax(Absent,0.001)), 
           bias = - (qnorm(pmin(Present,0.999))+qnorm(pmax(Absent,0.001)))/2) %>%
    group_by(!!!gp_vars[3:length(gp_vars)]) %>% # no 'sub' and 'target'
    nest()
  
  # grand average
  sdt_avg <- sdt_sub %>% unnest() %>%group_by(!!!gp_vars[3:length(gp_vars)]) %>%
    summarise(n = n(), mdprime = mean(dprime), dse = sd(dprime)/sqrt(n-1),
              mbias = mean(bias), cse = sd(bias)/sqrt(n-1)) 
  
  # return joined table
  sen = left_join(sdt_avg,sdt_sub)
  if ('reward' %in% colnames(df)){ # second experiment with reward
    # change labels, such that it differs from 'target' column
    sen$reward = factor(sen$reward, labels = c("RA","RP"))
  }
  return(sen)
}  

#' Linear regression of RT data
#' @param df individual sub data 
mrt_model <- function(df) {
  lm(rt ~ setsize, data = df)
}

# definition of fuction getSlope
getSlope <- function(df){
  
  # prepare group variables
  if ('reward' %in% colnames(df)){ # second experiment with reward
    gp_var = c("sub","target","dyn","reward")
  } else {
    gp_var = c("sub","target","dyn")
  }
  gp_vars = syms(gp_var) # prepare the group variables
  
  slopes <- df %>% group_by(sub) %>% 
    filter(correct == 1 & rt < mean(rt) + 2.5* sd(rt) & rt >0.1) %>% #remove incorrect and outliers
    group_by(!!!gp_vars) %>% nest()  %>%  # nested data
    mutate(model = map(data, mrt_model)) %>%  # linear regression
    mutate(slope = map(model, broom::tidy)) %>%  # get estimates out
    unnest(slope, .drop = TRUE) %>% # remove raw data
    select(-std.error,-statistic, -p.value) %>%  # remove unnessary clumns
    spread(term, estimate) %>%   # spread stimates
    rename(minRT = `(Intercept)`, slope = setsize)  # rename columns

  if ('reward' %in% colnames(df)){ # second experiment with reward
    # change labels, such that it differs from 'target' column
    slopes$reward = factor(slopes$reward, labels = c("RA","RP"))
  }
  
  return(slopes)
}

#' Get Mean Data
#' a function to get the mean RT, Error, etc. 
#' @param df experimental raw data
#' 
getMeanData <- function(df){
  # prepare group variables
  if ('reward' %in% colnames(df)){ # second experiment with reward
    gp_var = c("sub","setsize","target","dyn","reward")
  } else {
    gp_var = c("sub","setsize","target","dyn")
  }
  gp_vars = syms(gp_var) # prepare the group variables
  
  # 1. error analysis: mean errors for each subject
  error_sub = df %>% group_by(!!!gp_vars) %>%
    summarise(err = 1 - mean(correct)) %>%
    group_by(!!!gp_vars[2:length(gp_vars)]) %>% 
    nest() %>% rename(err_data = data) 
  # grand mean errors
  error = error_sub %>% unnest() %>%
    group_by(!!!gp_vars[2:length(gp_vars)]) %>%
    summarise(merr = mean(err), n = n(), se = sd(err)/sqrt(n-1)) %>%
    left_join(., error_sub)
  
  # 2. rt analysis: mean RTs with valid data 
  mrt_sub <- df %>% group_by(sub) %>% 
    filter(correct == 1 & rt < mean(rt) + 2.5* sd(rt) & rt >0.1) %>% #outlier
    group_by(!!!gp_vars) %>% #subject-wise
    summarise(mrt = mean(rt)) %>% 
    group_by(!!!gp_vars[2:length(gp_vars)]) %>% 
    nest() %>% rename(rt_data = data)
  
  # grand mean RTs
  mrt <- mrt_sub %>% unnest() %>%
    group_by(!!!gp_vars[2:length(gp_vars)]) %>% 
    summarise(mmrt = mean(mrt),  n = n(), sert = sd(mrt)/sqrt(n-1)) %>%
    left_join(., mrt_sub)
  
  # return joined table. 
  tbl = left_join(mrt,error)
  if ('reward' %in% colnames(df)){ # second experiment with reward
    # change labels, such that it differs from 'target' column
    tbl$reward = factor(tbl$reward, labels = c("RA","RP"))
  }
  return(tbl)
}

#' Plot RTs and Errors
#' plot mean RTs and errors
#' @param df table with mean data

plotRT <- function(df){
  
  if ("reward" %in% colnames(df)){
    fig_aes = aes(x = setsize, y = mmrt, color = target, 
                  shape = target, linetype = reward)
    lt = quo(reward)
  } else {
    fig_aes = aes(x = setsize, y = mmrt, color = target, shape = target, fill = target)
    lt = quo(target)
  }
  
  fig.rt <- ggplot(df, fig_aes) + 
    geom_point(size = 2) + geom_line() + 
    geom_errorbar(aes(ymin = mmrt - sert, ymax = mmrt + sert), width = 0.5) + 
    xlab('') + ylab('RT (Secs)') + 
    scale_x_continuous(breaks = c(8,12,16)) + 
    facet_wrap(~dyn) + 
    theme(legend.position = 'none', strip.background = element_blank())
  
  # change y axis
  fig_aes$y = quo(merr)
  
  fig.err <- ggplot(df, fig_aes) + 
    geom_bar(position = position_dodge(), stat = 'identity', fill = 'white') +
    geom_errorbar(aes(ymin = merr, ymax = merr + se), 
                  position = position_dodge()) + 
    xlab('Set Size') + ylab('Error rate') + 
    scale_x_continuous(breaks = c(8,12,16)) +
    facet_wrap(~dyn) +
    theme(legend.position = 'bottom', 
          strip.background = element_blank(), strip.text.x = element_blank())
  
  plot_grid(fig.rt,fig.err,nrow = 2, rel_heights = c(3,2))
}

#' Basic analysis
#' wrap up all basic average data, mean rt, error, sensitivity
#' @param df experimental raw data
#' 
basicEstimates <- function(df){
  mrt = getMeanData(df)  # mean RT and errors
  slope = getSlope(df)  # slopes
  sensitivity = sdt_dprime(df) # sensitivity and bias
  return(list('mrt' =mrt, 'slope' = slope, 'sensitivity' = sensitivity))
}


#' Plot slope bars
#' plotting out bars and errorbar of mean slopes
#' @param df a table of slope
plotSlopeErrbar <- function(df){
  if ('reward' %in% colnames(df)){ # second experiment with reward
    gp_var = c("target","dyn","reward")
    fig_aes = aes(x = dyn, y = m_slope, ymin = m_slope, color = interaction(target,reward), ymax = m_slope + sd_slope,
                  group = interaction(target,reward), fill = interaction(target,reward))
  } else {
    gp_var = c("target","dyn")
    fig_aes = aes(x = dyn, y = m_slope, ymin = m_slope, color = target, ymax = m_slope + sd_slope,
                  group = target, fill = target)
  }
  gp_vars = syms(gp_var) # prepare the group variables
  
  mslope <- df %>% group_by(!!!gp_vars) %>% 
    summarise(m_slope = mean(slope)*1000, n = n(), sd_slope = sd(slope)/ sqrt(n-1)*1000)
  
  fig.mSlope.bar <- ggplot(mslope, fig_aes) + 
    geom_bar(stat = "identity",
             position = position_dodge()) +
    geom_errorbar(width = 0.2,
                  position = position_dodge(width = 0.9))+ 
    labs(x = "Display Type", y = "Slope (ms/item)") +
    theme(legend.position = 'bottom', legend.title = element_blank())
  
  return(fig.mSlope.bar)
}


# ---- some functions from Fiona ---

# definition of fuction computing rt pro item
# input df: mrt of experiment
plotRTPI <- function(df){
  mrt_P <- df %>% filter(target == "Present") 
  mrt_A <- df %>% filter(target == "Absent")
  mrt_J <- left_join(x = mrt_P, y = mrt_A, by = c("setsize","dyn"), all.x = TRUE)
  mrt_J <- mrt_J[c('setsize', 'mmrt.x','mmrt.y', 'dyn')]
  
  rtpi <-  {}
  rt_motor <-  {}
  for(row in 1:nrow(mrt_J)){
    dat <- mrt_J[row,]
    #definition of model for solution of rtpi (x[1]: rt pro item) and RT_{motor}(x[2])
    #variables:  n = dat["setsize"] the number of item
    #            RT_{P} = dat["mmrt.x"]  the reaction time when target == Present
    #            RT_{A} = dat["mmrt.y"]  the reaction time when target == Absent
    nleqslv <- function(x){
      r1 <- x[1]*dat["setsize"]/2 + x[2] - dat["mmrt.x"]
      r2 <- x[1]*dat["setsize"] + x[2] - dat["mmrt.y"]
      r
    }
    lf<-matrix(c(dat[1][[1]]/2, 1, dat[1][[1]], 1), nrow = 2,byrow = TRUE)
    rf<-matrix(c(dat[2][[1]], dat[3][[1]]),nrow = 2)
    result<-solve(lf,rf)
    
    rtpi <- rbind2(rtpi, result[1][1])
    rt_motor <-  rbind2(rt_motor, result[2][1])
  } 
  mrt_J["rtpi"] = rtpi
  mrt_J["rtmotor"] = rt_motor
  
  fig.rtpi <- ggplot(mrt_J, aes(x = setsize, y = rtpi, color = dyn)) + 
    geom_point(size = 2)+ 
    geom_line()
  
  fig.rtmotor <- ggplot(mrt_J, aes(x = setsize, y = rtmotor, color = dyn)) + 
    geom_point(size = 2)+ 
    geom_line()
  
  #return(list("fig.rtpi" = fig.rtpi, "fig.rtmotor" = fig.rtmotor))
  return(plot_grid(fig.rtpi, fig.rtmotor, ncol = 1, nrow = 2))
}



#definition of fuction to display the reaction times histogratm of the 'correct'/'error' responses 
# input df: dataframe to handle
# input is_correct: 0  'error' responses
#                   1  'correct' responses 
histRT <- function(df, is_correct){
  fig.histRT <- df %>%  filter(dyn == 'Dynamic' & rt < 4.95 & correct == is_correct) %>% 
    ggplot(aes(rt)) + geom_histogram() + 
    facet_grid(target ~ setsize)
  return(fig.histRT)
}

# definition of fuction to display the reaction times histogratm of 
# the 'correct'/'error' responses for the expriment with rewards
# input df: dataframe to handle
# input is_correct: 0  'error' responses
#                   1  'correct' responses 
histRT_rewards <- function(df, is_correct){
  
  histRT <- df %>%  filter(dyn == 'Dynamic' & rt < 4.95 & correct == is_correct) 
  levels(histRT$reward) <- c('Reward Ab.', 'Reward Pr.')
  
  fig.histRT <- ggplot(histRT, aes(rt)) + geom_histogram() + 
    facet_grid(target+reward ~ setsize) 
  return(fig.histRT)
}



# Definition of fuction for a box plot and calculate the significance of a difference between groups
plotSlopebox <- function(df){
  anno_df = compare_means(slope ~ target, group.by = "dyn", data = df) %>%
    mutate(y_pos = 0.19)
  
  fig.mSlope.box <- ggboxplot(df, x="target", y="slope", color ="target", palette = "jco",
                              add = "jitter",
                              facet.by = "dyn", short.panel.labs = FALSE )+
    ggsignif::geom_signif(
      data=anno_df, 
      aes(xmin=group1, xmax=group2, annotations=p.signif, y_position=y_pos), 
      manual=TRUE
    )
  
  return(fig.mSlope.box)
}

#' plot dprime and bias
#' @param df table of sensitivity
#' 
plotSDT <- function(df){
  
  if ("reward" %in% colnames(df)){
    fig_aes = aes(x = setsize, y = mdprime, color = dyn, 
                  shape = dyn, linetype = reward)
  } else {
    fig_aes = aes(x = setsize, y = mdprime, color = dyn, shape = dyn, fill = dyn)
  }
  
  err_aes = aes(ymin = mdprime - dse, ymax = mdprime + dse)
  
  fig.dprime <- ggplot(df, fig_aes) + 
    geom_point(size = 2) + geom_line() + 
    geom_errorbar(err_aes, width = 0.5) + 
    xlab('') + ylab("d'") + 
    scale_x_continuous(breaks = c(8,12,16)) + 
    theme(legend.position = 'bottom', legend.title = element_blank())
  
  # plot biase, by changing y axis
  fig_aes$y = quo(mbias)
  err_aes = aes(ymin = mbias - cse, ymax = mbias + cse)
  
  fig.bias <- ggplot(df, fig_aes) + 
    geom_point(size = 2) + geom_line() + 
    geom_errorbar(err_aes, width = 0.5) + 
    xlab('') + ylab("C") + 
    scale_x_continuous(breaks = c(8,12,16)) + 
    theme(legend.position = 'bottom', legend.title = element_blank())
  
  return(list('dprime' = fig.dprime, 'bias' = fig.bias))
}

# parallel processing of estimates. 
parEstimate <- function(fun, para){
  no_cores <- min(detectCores() - 1, 10)
  print(no_cores)
  cl <- makeCluster(no_cores)
  clusterEvalQ(cl, {library(dplyr) })
  clusterExport(cl, c('simulate_trial', 'simulate_trial2', 'simulate_trial3', 
                      'simulate_experiment', 'fit_model', 'e1_data'))
  
  t0 = proc.time()
  ll <- clusterMap(cl, fun, para)
  stopCluster(cl)
  print(proc.time()-t0)
  return(ll)
}
