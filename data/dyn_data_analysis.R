# ---- init_load ----
## initialize library and load files, as well as load functions 
# load analysis functions: 
#   getMeanData, getSlope, plotRT, sdt_dprime, basicEstimates
source('ana_functions.R')

# read in 4 experimental datasets
# see also readme.pdf in experiments folder
# Exp. 1. Replication of Horowitz & Wolfe (1998) Exp. 1
# Exp. 2. reward manipulation: session wise (two sessions/exps)
# Exp. 3. control the display density to equal, with error feedback. 
#     small setsize display will be allocated to different quant. of display (location)
# Exp. 5. task difficulty increased
exp_names = c('exp1','exp2','exp3','exp5')
exp_files = paste0(exp_names,'.csv')
exps = map(exp_files, read.csv)
dat = map(exps, basicEstimates)
names(dat) <- exp_names

# flag for save figures
saveFigure = FALSE

# ---- rt_figures ----
# plot rt figures and save to the subfolder figures
figRTs = list()
for (exname in exp_names) {
  figRTs[[exname]] = plotRT(dat[[exname]]$mrt)
  if (saveFigure == TRUE)
    ggsave(file.path('figures',paste0(exname,'_rt.png')), figRTs[[exname]] ,width = 7, height = 5)
}


# ---- slope_sdt_figures ---
figSlopes = list()
fig_ds = list()
fig_cs = list()
for (exname in exp_names) {
  figSlopes[[exname]] = plotSlopeErrbar(dat[[exname]]$slope)
  fig = plotSDT(dat[[exname]]$sensitivity)
  fig_ds[[exname]] = fig$dprime
  fig_cs[[exname]] = fig$bias
}

# ---- combined_plot ---
fig_combine = list()
for (exname in exp_names) {
  fig = plot_grid(figSlopes[[exname]],fig_ds[[exname]],fig_cs[[exname]],nrow = 1, labels = c("A","B","C"))
  if (saveFigure == TRUE) {
    ggsave(file.path('figures',paste0(exname,'_combine.png')), fig ,width = 9, height = 3)
  }
  fig_combine[[exname]] = fig
}

# ---- ANOVA for RT ----
rt_anova = list()
for (exname in exp_names) {
  rt_anova[[exname]] = dat[[exname]]$mrt %>% unnest() %>% ungroup() %>%
    mutate_at(c('sub','setsize'), funs(factor(.))) %>% # convert to factors
    { if(exname == 'exp2') 
        ezANOVA(data = ., dv = mrt, wid = sub,
                within = .(setsize, target, dyn, reward))
      else
        ezANOVA(data = ., dv = mrt, wid = sub,
                within = .(setsize, target, dyn))
    }  
}

# ---- slope_anova ----
slope_anova = list()
for (exname in exp_names) {
  slope_anova[[exname]] = dat[[exname]]$slope %>% ungroup() %>%
    mutate_at(c('sub'), funs(factor(.))) %>% # convert to factors
    { if(exname == 'exp2') 
      ezANOVA(data = ., dv = slope, wid = sub,
              within = .(target, dyn, reward))
      else
        ezANOVA(data = ., dv = slope, wid = sub,
                within = .(target, dyn))
    }
}

# ---- slope_anova ----
d_anova = list() # anova for d'
c_anova = list() # anova for c
for (exname in exp_names) {
  d_anova[[exname]] = dat[[exname]]$sensitivity %>% unnest() %>% ungroup() %>%
    mutate_at(c('sub','setsize'), funs(factor(.))) %>% # convert to factors
    { if(exname == 'exp2') 
      ezANOVA(data = ., dv = dprime, wid = sub,
              within = .(dyn, setsize,reward))
      else
        ezANOVA(data = ., dv = dprime, wid = sub,
                within = .(dyn,setsize))
    }
  c_anova[[exname]] = dat[[exname]]$sensitivity %>% unnest() %>% ungroup() %>%
    mutate_at(c('sub','setsize'), funs(factor(.))) %>% # convert to factors
    { if(exname == 'exp2') 
      ezANOVA(data = ., dv = bias, wid = sub,
              within = .(dyn, setsize, reward))
      else
        ezANOVA(data = ., dv = bias, wid = sub,
                within = .( dyn, setsize))
    }}


#---- old_code ----

if (FALSE) {
#calculate slope for Exp2
#nested data frame
#exp2 %>% group_by(sub) %>% 
#  filter(correct == 1 & rt < mean(rt) + 2.5* sd(rt) & rt >0.1) %>% 
#  group_by(sub, target, dyn, reward) %>% nest() -> mrt_nt2

#define a model-fitting function
mrt2_model <- function(df) {
  lm(rt ~ setsize, data = df)
}

models <- map(mrt_nt2$data, mrt2_model) 
mrt_nt2 <- mrt_nt2 %>% 
  mutate(model = map(data, mrt2_model))

#computed the slope and unnesting
slope2 <- mrt_nt2 %>% 
  mutate(slope = map(model, broom::tidy)) %>% 
  unnest(slope, .drop = TRUE) %>% 
  filter(term == "setsize")

#reconstruct the dataframe and save
slope2  <- slope2[ , c("sub", "target", "dyn", "reward", "estimate")]
slope2["slope"] = slope2["estimate"]
slope2["estimate"] <- NULL

# run Repeated Measures ANOVA
mod2.aov  <- aov(slope ~ (target*reward*dyn) + Error(sub/(target*reward)), slope2 %>% filter(!(target == "Present" & dyn == "Static")))
summary(mod2.aov)


# run ezANOVA
mod2.ez <- ezANOVA(slope2, dv=slope, wid=sub, within=.(target, reward), between=.(dyn))
mod2.ez



fig2.slope.Errbar <- plotSlopeErrbar(slope2)
fig2.slope.Errbar

mslope2 <- slope2 %>% group_by(sub) %>% 
  group_by(target, dyn, reward) %>% 
  summarise(m_slope = mean(slope), n = n(), sd_slope = sd(slope)/ sqrt(n-1))



fig2.slope.Errbar <- ggplot(mslope2, aes(target, m_slope, ymin = m_slope-sd_slope, 
                                     ymax = m_slope + sd_slope, group = dyn,reward, fill = dyn)) + 
  geom_bar(stat = "identity",
           color = "black",
           position = position_dodge()) +
  geom_errorbar(width = 0.2,
                position = position_dodge(width = 0.9))+ 
  facet_wrap(~reward)+
  labs(x = "Target", y = "Mean Slope", size =5)



anno_df2 = compare_means(slope ~ target , group.by = c("dyn", "reward"), data = slope2) %>%
  mutate(y_pos = 0.12)

facet.by = "dyn"
fig2.slope.box <- ggboxplot(slope2, x="target", y="slope", color ="reward", palette = "jco",
                            add = "jitter", group=c("dyn"),
                            short.panel.labs = FALSE )+
  facet_wrap(dyn~. )+
  ggsignif::geom_signif(
    data=anno_df2, 
    aes(xmin=group1, xmax=group2, annotations=p.signif, y_position=y_pos), 
    manual=TRUE
  )
fig2.slope.box 



#---- Exp3 ----
plotErr3 <- plotErr(exp3) #plot error
fig3.err <- plotErr3$fig.err
fig3.err

#method 1 #TODO to check
plotRT3 <- plotRT(exp3)  #reaction time
fig3.rt <- plotRT1$fig.rt
fig3.rt
plot_grid(fig3.rt,fig3.err, ncol = 1, nrow = 2)

#method 2
error3 <- plotErr3$error
mrt3 <- plotRT3$mrt
#replace the dataset
fig3.rt <- fig1.rt %+% mrt3
fig3.err <- fig1.err %+% error3 
plot_grid(fig3.rt,fig3.err, ncol = 1, nrow = 2)

#plot rt pro item
fig3.rtpi <- plotRTPI(mrt3)
fig3.rtpi

#calculate slope for Exp3
slope3 <- getSlope(exp3)

#  run Repeated Measures ANOVA
mod3.aov <- aov(slope ~ target*dyn+ Error(sub/(target)), 
            slope3 %>% filter(!(target == "Present" & dyn == "Static")))
summary(mod3.aov )

# run ezANOVA
mod3.ez <- ezANOVA(slope3, dv=slope, wid=sub, within=.(target), between=.(dyn))
mod3.ez



fig3.slope.Errbar <- plotSlopeErrbar(slope3)
fig3.slope.Errbar

fig3.slope.box <- plotSlopebox(slope3)
fig3.slope.box

# ---- Exp5 ----
#exp5$target = factor(exp5$target, labels = c("Present","Absent"))
#exp5$dyn = factor(exp5$dyn, labels = c("Dynamic","Static"))
#exp5$setsize = factor(exp5$setsize, labels = c("8","12", "16"))
#write.csv(exp5, file = 'data/exp5.csv') 


plotErr5 = plotErr(exp5)
error5 <- plotErr5$error
fig5.err = plotErr5$fig.err
fig5.err

plotRT5 = plotRT(exp5)
mrt5 <-plotRT5$mrt
fig5.rt <- plotRT5$fig.rt
fig5.rt

#plot rt pro item
fig5.rtpi <- plotRTPI(mrt5)
fig5.rtpi


#calculate slope
slope5 <- getSlope(exp5)

# run Repeated Measures ANOVA
mod5.aov <- aov(slope ~ target * dyn+ Error(sub/(target)), 
                slope5 %>% filter(!(target == "Present" & dyn == "Static")))
summary(mod5.aov)


# run ezANOVA
mod5.ez <- ezANOVA(slope5, dv=slope, wid=sub, within=.(target), between=.(dyn))
mod5.ez

# TODO run Bayesian Repeated Measures ANOVA



fig5.slope.Errbar <- plotSlopeErrbar(slope5)
fig5.slope.Errbar

fig5.slope.box <- plotSlopebox(slope5)
fig5.slope.box


# write to csv multiple sheets
write.csv(error1, file = 'data/exp1_err.csv')
write.csv(error2, file = 'data/exp2_err.csv')
write.csv(error3, file = 'data/exp3_err.csv')
write.csv(error5, file = 'data/exp5_err.csv')

write.csv(mrt1, file = 'data/exp1_mrt.csv')
write.csv(mrt2, file = 'data/exp2_mrt.csv')
write.csv(mrt3, file = 'data/exp3_mrt.csv')
write.csv(mrt5, file = 'data/exp5_mrt.csv')

# save slope data
write.csv(slope1, file = 'data/slope1.csv')  
write.csv(slope2, file = 'data/slope2.csv')  
write.csv(slope3, file = 'data/slope3.csv')  
write.csv(slope5, file = 'data/slope5.csv') 


}
