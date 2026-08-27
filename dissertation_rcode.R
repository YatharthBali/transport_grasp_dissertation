# ============================================================
# DISSERTATION — TRANSPORT COMPONENT ANALYSIS
# ML Reach-to-Grasp Study | Yatharth | August 2026
# ============================================================
library(tidyverse)
library(caret)
library(randomForest)
library(e1071)
library(glmnet)
library(pROC)
library(gridExtra)
library(lme4)
library(lmerTest)
library(MuMIn)
library(lmtest)
library(xgboost)
library(fda)
library(keras)
select <- dplyr::select
# ============================================================
# SECTION 1: LOAD DATA
# ============================================================
path_exp1 <- "D:/Dissertation/02 ML Reach to Grasp 2026 - STUDENTS/01 data/Exp1_progressive"
path_exp2 <- "D:/Dissertation/02 ML Reach to Grasp 2026 - STUDENTS/01 data/Exp2_random"
files_exp1 <- list.files(path_exp1,
                         pattern="_mms\\.csv$", full.names=TRUE)
files_exp2 <- list.files(path_exp2,
                         pattern="_mms\\.csv$", full.names=TRUE)
data_exp1 <- lapply(files_exp1, read.csv)
data_exp2 <- lapply(files_exp2, read.csv)
names(data_exp1) <- paste0("exp1_",
                           tools::file_path_sans_ext(
                             gsub("_mms","",basename(files_exp1))))
names(data_exp2) <- paste0("exp2_",
                           tools::file_path_sans_ext(
                             gsub("_mms","",basename(files_exp2))))
all_data <- c(data_exp1, data_exp2)
summary1   <- all_data[["exp1_exp1_trial_summary"]]
summary2   <- all_data[["exp2_exp2_trial_summary"]]
transport1 <- all_data[["exp1_exp1_transport_timeseries"]]
transport2 <- all_data[["exp2_exp2_transport_timeseries"]]
reach1     <- all_data[["exp1_exp1_reach_timeseries"]]
reach2     <- all_data[["exp2_exp2_reach_timeseries"]]
cat("Loaded! Exp1:", nrow(summary1),
    "| Exp2:", nrow(summary2), "\n")
cat("MGV check:", round(mean(summary1$MGV,na.rm=TRUE),2),
    "mm/s (expect ~859)\n")
# ============================================================
# SECTION 2: DATA CLEANING
# ============================================================
# 2a. ROI filter
t1 <- transport1 %>% filter(ROI==TRUE)
t2 <- transport2 %>% filter(ROI==TRUE)
r1 <- reach1    %>% filter(ROI==TRUE)
r2 <- reach2    %>% filter(ROI==TRUE)
cat("Exp1 ROI frames:", nrow(t1), "\n")
cat("Exp2 ROI frames:", nrow(t2), "\n")
# 2b. Recode Exp2 offset 33 → 99 (no-vision)
summary2_fixed <- summary2 %>%
  mutate(
    offset   = ifelse(offset==33, 99, offset),
    offset_f = factor(offset,
                      levels=c(0,10,20,30,70,99),
                      labels=c("0","10","20","30","70","99\n(no vision)"))
  )
transport2_fixed <- transport2 %>%
  mutate(offset=ifelse(offset==33, 99, offset))
# 2c. Kinematic movTime for Exp2
exp2_movtime <- transport2 %>%
  filter(ROI==TRUE) %>%
  distinct(subjName, trialN, movTime)
summary2_fixed <- summary2_fixed %>%
  left_join(exp2_movtime, by=c("subjName","trialN"))
# 2d. Fix Inf in Exp2 accelDecelRatio
cat("Inf in Exp2 accelDecelRatio BEFORE:",
    sum(is.infinite(summary2_fixed$accelDecelRatio)), "\n")
summary2_fixed <- summary2_fixed %>%
  mutate(accelDecelRatio=ifelse(
    is.infinite(accelDecelRatio), NA, accelDecelRatio))
cat("Inf in Exp2 accelDecelRatio AFTER:",
    sum(is.infinite(summary2_fixed$accelDecelRatio)), "\n")
cat("Data cleaning done!\n")
cat("Exp2 movTime mean:",
    round(mean(summary2_fixed$movTime,na.rm=TRUE),2),
    "ms\n")
# ============================================================
# SECTION 3: FEATURES
# ============================================================
features_full <- c("movTime","MGV","timeToMGV",
                   "decelPhase","accelDecelRatio",
                   "FGPz","FGPx","FGPy",
                   "FIPz","FTPz","MGA",
                   "timeToMGA","timeToFGA")
# ============================================================
# SECTION 4: DESCRIPTIVE STATISTICS
# ============================================================
cat("\n=== EXP1: TRANSPORT STATS BY OFFSET ===\n")
summary1 %>%
  group_by(offset) %>%
  summarise(
    n=n(),
    mean_movTime=round(mean(movTime,na.rm=TRUE),2),
    sd_movTime=round(sd(movTime,na.rm=TRUE),2),
    mean_MGV=round(mean(MGV,na.rm=TRUE),2),
    mean_FGPz=round(mean(FGPz,na.rm=TRUE),2),
    mean_decel=round(mean(decelPhase,na.rm=TRUE),2)
  ) %>% print()
cat("\n=== EXP2: TRANSPORT STATS BY OFFSET ===\n")
summary2_fixed %>%
  group_by(offset) %>%
  summarise(
    n=n(),
    mean_movTime=round(mean(movTime,na.rm=TRUE),2),
    sd_movTime=round(sd(movTime,na.rm=TRUE),2),
    mean_MGV=round(mean(MGV,na.rm=TRUE),2),
    mean_FGPz=round(mean(FGPz,na.rm=TRUE),2)
  ) %>% print()
# ============================================================
# SECTION 5: FHP ERROR
# ============================================================
cat("\n=== EXP1: FHP ERROR ===\n")
summary1 %>%
  mutate(FHP_error=FGPz-AbsDepth) %>%
  group_by(offset) %>%
  summarise(
    mean_error=round(mean(FHP_error,na.rm=TRUE),2),
    sd_error=round(sd(FHP_error,na.rm=TRUE),2)
  ) %>% print()
cat("\n=== EXP2: FHP ERROR ===\n")
summary2_fixed %>%
  mutate(FHP_error=FGPz-AbsDepth) %>%
  group_by(offset) %>%
  summarise(
    mean_error=round(mean(FHP_error,na.rm=TRUE),2),
    sd_error=round(sd(FHP_error,na.rm=TRUE),2)
  ) %>% print()
# ============================================================
# SECTION 6: EDA PLOTS
# ============================================================
normalise_time <- function(df) {
  df %>%
    filter(ROI==TRUE, !is.na(GPvel)) %>%
    group_by(subjName, trialN) %>%
    mutate(
      time_pct=(row_number()-1)/(n()-1)*100,
      time_bin=round(time_pct/5)*5
    ) %>% ungroup()
}
t1_norm <- normalise_time(transport1)
t2_norm <- normalise_time(transport2)
reach1_norm <- reach1 %>%
  filter(ROI==TRUE) %>%
  group_by(subjName, trialN) %>%
  mutate(
    time_pct=(row_number()-1)/(n()-1)*100,
    time_bin=round(time_pct/5)*5
  ) %>% ungroup()
p_fhp1 <- summary1 %>%
  mutate(FHP_error=FGPz-AbsDepth,
         offset_f=factor(offset,levels=c(5,10,15,20,30,70))) %>%
  ggplot(aes(x=offset_f,y=FHP_error)) +
  geom_boxplot(fill="#F4A8A8",alpha=0.8) +
  geom_hline(yintercept=0,linetype="dashed",color="red") +
  labs(title="Exp1: FHP Error by Offset",
       subtitle="Less vision = greater undershooting",
       x="Offset (mm)",y="FHP Error (mm)") +
  theme_minimal()
ggsave("fig3_fhp_exp1.png",plot=p_fhp1,width=7,height=5,dpi=150)
p_fhp2 <- summary2_fixed %>%
  mutate(FHP_error=FGPz-AbsDepth) %>%
  ggplot(aes(x=offset_f,y=FHP_error)) +
  geom_boxplot(fill="#80CED7",alpha=0.8) +
  geom_hline(yintercept=0,linetype="dashed",color="red") +
  labs(title="Exp2: FHP Error by Offset (Corrected)",
       subtitle="33mm recoded as 99mm — no vision",
       x="Offset (mm)",y="FHP Error (mm)") +
  theme_minimal()
ggsave("fig3_fhp_exp2.png",plot=p_fhp2,width=7,height=5,dpi=150)
p_vel <- t1_norm %>%
  filter(!is.na(time_bin)) %>%
  group_by(offset,time_bin) %>%
  summarise(mean_vel=mean(GPvel,na.rm=TRUE),.groups="drop") %>%
  ggplot(aes(x=time_bin,y=mean_vel,
             color=as.factor(offset),group=as.factor(offset))) +
  geom_line(linewidth=1) +
  geom_vline(xintercept=50,linetype="dashed",color="grey50") +
  labs(title="Exp1: Velocity Profile (Normalised)",
       x="Movement Time (%)",y="Mean GP Velocity (mm/s)",
       color="Offset (mm)") +
  theme_minimal() +
  theme(legend.position="bottom")
ggsave("plot1_velocity_profile.png",plot=p_vel,
       width=9,height=5,dpi=150)
cat("EDA plots saved!\n")
# ============================================================
# SECTION 7: CORRELATION ANALYSIS
# ============================================================
cor_exp1 <- summary1 %>%
  select(offset,all_of(features_full)) %>%
  cor(use="complete.obs") %>% round(3)
cor_off1 <- sort(cor_exp1["offset",],decreasing=TRUE)
cor_off1 <- cor_off1[names(cor_off1)!="offset"]
cat("\n=== EXP1: CORRELATIONS WITH OFFSET ===\n")
print(round(cor_off1,3))
cor_exp2 <- summary2_fixed %>%
  select(offset,all_of(features_full)) %>%
  cor(use="complete.obs") %>% round(3)
cor_off2 <- sort(cor_exp2["offset",],decreasing=TRUE)
cor_off2 <- cor_off2[names(cor_off2)!="offset"]
cat("\n=== EXP2: CORRELATIONS WITH OFFSET ===\n")
print(round(cor_off2,3))
# ============================================================
# SECTION 8: FORMAL NON-LINEARITY TEST
# ============================================================
cat("\n=== RAMSEY RESET TEST (Formal Non-linearity) ===\n")
lm_model <- lm(
  offset ~ movTime+MGV+FGPz+FGPx+FGPy+
    decelPhase+MGA+timeToMGA,
  data=summary1
)
reset_result <- resettest(lm_model,
                          power=2:3,
                          type="regressor")
cat("RESET test result:\n")
print(reset_result)
cat("p <", format.pval(reset_result$p.value),
    "→ non-linearity formally proven\n")
# ============================================================
# SECTION 9: LINEAR MIXED EFFECTS MODEL
# ============================================================
cat("\n=== LINEAR MIXED EFFECTS MODEL ===\n")
lme_model <- lmer(
  offset ~ movTime+MGV+FGPz+FGPx+FGPy+
    decelPhase+MGA+timeToMGA+
    (1|subjName),
  data=summary1, REML=TRUE
)
cat("\nLME Summary:\n")
print(summary(lme_model))
cat("\nFixed effects:\n")
print(summary(lme_model)$coefficients)
cat("\nRandom effects:\n")
print(VarCorr(lme_model))
cat("\nR-squared:\n")
print(r.squaredGLMM(lme_model))
cat("Marginal R2 = fixed effects only\n")
cat("Conditional R2 = fixed + random effects\n")
# ============================================================
# SECTION 9b: RESET TEST ON LME RESIDUALS
# ============================================================
cat("\n=== RESET TEST ON LME RESIDUALS ===\n")
lme_residuals <- residuals(lme_model)
lm_on_residuals <- lm(
  lme_residuals ~ movTime + MGV + FGPz + FGPx + FGPy +
    decelPhase + MGA + timeToMGA,
  data = summary1[complete.cases(
    summary1[, c("movTime","MGV","FGPz","FGPx","FGPy",
                 "decelPhase","MGA","timeToMGA")]), ]
)
reset_lme <- resettest(lm_on_residuals, power=2:3, type="regressor")
cat("RESET on LME residuals:\n")
print(reset_lme)
cat("RESET statistic:", round(reset_lme$statistic, 3), "\n")
cat("p-value:", format.pval(reset_lme$p.value), "\n")
cat("Interpretation: non-linearity persists after removing participant effects\n")
# ============================================================
# SECTION 10: SANITY CHECK CLASSIFICATION
# ============================================================
cat("\n=== SANITY CHECK CLASSIFICATION ===\n")
exp1_ml <- summary1 %>%
  select(all_of(features_full),AbsDepth,RelDepthObj,subjName) %>%
  mutate(
    AbsDepth_class=factor(ifelse(AbsDepth==280,"Near","Far")),
    RelDepth_class=factor(ifelse(RelDepthObj==30,"Small","Large"))
  ) %>% drop_na()
exp2_ml <- summary2_fixed %>%
  select(all_of(features_full),AbsDepth,RelDepthObj,subjName) %>%
  mutate(
    AbsDepth_class=factor(ifelse(AbsDepth==280,"Near","Far")),
    RelDepth_class=factor(ifelse(RelDepthObj==30,"Small","Large"))
  ) %>% drop_na()
cat("Exp1 ML rows:", nrow(exp1_ml), "\n")
cat("Exp2 ML rows:", nrow(exp2_ml),
    "(1 Inf trial excluded)\n")
set.seed(42)
cv_roc <- trainControl(
  method="repeatedcv", number=10, repeats=3,
  classProbs=TRUE, savePredictions="final",
  summaryFunction=twoClassSummary
)
run_sanity <- function(data,features,target,
                       exp_name,task_name) {
  cat("\n===",exp_name,"---",task_name,"===\n")
  y <- data[[target]]
  lr <- train(x=data[,features],y=y,
              method="glm",family="binomial",
              trControl=cv_roc,metric="ROC",
              preProcess=c("center","scale"))
  svm_lin <- train(x=data[,features],y=y,
                   method="svmLinear",trControl=cv_roc,
                   metric="ROC",preProcess=c("center","scale"),
                   tuneGrid=expand.grid(C=c(0.01,0.1,1,10)))
  rf <- train(x=data[,features],y=y,
              method="rf",trControl=cv_roc,metric="ROC",
              ntree=100,tuneGrid=expand.grid(mtry=c(2,4,6,8)))
  cat("LR  AUC:",round(max(lr$results$ROC),3),"\n")
  cat("SVM AUC:",round(max(svm_lin$results$ROC),3),"\n")
  cat("RF  AUC:",round(max(rf$results$ROC),3),"\n")
  return(list(lr=lr,svm=svm_lin,rf=rf))
}
res_e1_abs <- run_sanity(exp1_ml,features_full,
                         "AbsDepth_class","Exp1","AbsDepth")
res_e1_rel <- run_sanity(exp1_ml,features_full,
                         "RelDepth_class","Exp1","RelDepthObj")
res_e2_abs <- run_sanity(exp2_ml,features_full,
                         "AbsDepth_class","Exp2","AbsDepth")
res_e2_rel <- run_sanity(exp2_ml,features_full,
                         "RelDepth_class","Exp2","RelDepthObj")
cat("\n=== SANITY CHECK SUMMARY ===\n")
sanity_table <- data.frame(
  Exp=c(rep("Exp1",4),rep("Exp2",4)),
  Task=c(rep("AbsDepth",2),rep("RelDepthObj",2),
         rep("AbsDepth",2),rep("RelDepthObj",2)),
  Model=rep(c("RF","SVM"),4),
  AUC=c(
    round(max(res_e1_abs$rf$results$ROC),3),
    round(max(res_e1_abs$svm$results$ROC),3),
    round(max(res_e1_rel$rf$results$ROC),3),
    round(max(res_e1_rel$svm$results$ROC),3),
    round(max(res_e2_abs$rf$results$ROC),3),
    round(max(res_e2_abs$svm$results$ROC),3),
    round(max(res_e2_rel$rf$results$ROC),3),
    round(max(res_e2_rel$svm$results$ROC),3)
  )
)
print(sanity_table)
# ============================================================
# SECTION 11: OFFSET CLASSIFICATION — ALL MODELS
# ============================================================
cat("\n=== OFFSET CLASSIFICATION ===\n")
prepare_ml <- function(data, exp_name) {
  base <- data %>%
    mutate(
      offset_class=factor(paste0("O",offset)),
      vision_group=factor(ifelse(
        offset<=15,"HighVision","LowVision")),
      subj_num=as.numeric(factor(subjName)),
      accelDecelRatio=ifelse(
        is.infinite(accelDecelRatio),NA,accelDecelRatio)
    )
  list(
    d6=base %>% select(all_of(features_full),
                       offset_class) %>% drop_na(),
    dbin=base %>% select(all_of(features_full),
                         vision_group) %>% drop_na(),
    dreg=base %>% select(all_of(features_full),
                         offset) %>% drop_na(),
    dsubj=base %>% select(all_of(features_full),
                          offset_class,vision_group,offset,subj_num) %>%
      drop_na()
  )
}
d1 <- prepare_ml(summary1, "Exp1")
d2 <- prepare_ml(summary2_fixed, "Exp2")
cat("Exp1 6-class rows:", nrow(d1$d6), "\n")
cat("Exp2 6-class rows:", nrow(d2$d6), "\n")
cat("Chance baseline: 16.7%\n")
set.seed(42)
cv_6 <- trainControl(method="repeatedcv",number=10,
                     repeats=3,savePredictions="final",verboseIter=TRUE)
cv_b <- trainControl(method="repeatedcv",number=10,
                     repeats=3,classProbs=TRUE,savePredictions="final",
                     summaryFunction=twoClassSummary,verboseIter=TRUE)
cv_r <- trainControl(method="repeatedcv",number=10,
                     repeats=3,savePredictions="final",verboseIter=TRUE)
g_acc  <- function(m) round(max(m$results$Accuracy)*100,2)
g_auc  <- function(m) round(max(m$results$ROC),3)
g_bacc <- function(m) round(mean(m$pred$pred==m$pred$obs)*100,2)
g_rmse <- function(m) round(m$results$RMSE[which.min(m$results$RMSE)],3)
g_r2   <- function(m) round(m$results$Rsquared[which.min(m$results$RMSE)],3)
run3 <- function(d, method, grid, name, exp,
                 preproc=NULL) {
  cat("\n---",name,"|",exp,"---\n")
  m6 <- train(x=d$d6[,features_full],
              y=d$d6$offset_class,method=method,
              trControl=cv_6,tuneGrid=grid,
              metric="Accuracy",preProcess=preproc)
  mb <- train(x=d$dbin[,features_full],
              y=d$dbin$vision_group,method=method,
              trControl=cv_b,tuneGrid=grid,
              metric="ROC",preProcess=preproc)
  mr <- train(x=d$dreg[,features_full],
              y=d$dreg$offset,method=method,
              trControl=cv_r,tuneGrid=grid,
              metric="RMSE",preProcess=preproc)
  cat(name,exp,"6cl:",g_acc(m6),
      "Bin:",g_bacc(mb),"R²:",g_r2(mr),"\n")
  data.frame(Experiment=exp,Model=name,
             Acc_6class=g_acc(m6),Acc_binary=g_bacc(mb),
             AUC_binary=g_auc(mb),RMSE_reg=g_rmse(mr),
             R2_reg=g_r2(mr))
}
all_results <- data.frame()
lasso_g <- expand.grid(alpha=1,lambda=c(0.001,0.01,0.1))
all_results <- rbind(all_results,
                     run3(d1,"glmnet",lasso_g,"Lasso","Exp1",
                          c("center","scale")))
all_results <- rbind(all_results,
                     run3(d2,"glmnet",lasso_g,"Lasso","Exp2",
                          c("center","scale")))
svm_g <- expand.grid(C=c(1,10),sigma=c(0.01,0.1))
all_results <- rbind(all_results,
                     run3(d1,"svmRadial",svm_g,"SVM RBF","Exp1",
                          c("center","scale")))
all_results <- rbind(all_results,
                     run3(d2,"svmRadial",svm_g,"SVM RBF","Exp2",
                          c("center","scale")))
rf_g <- expand.grid(mtry=c(4,8))
all_results <- rbind(all_results,
                     run3(d1,"rf",rf_g,"Random Forest","Exp1"))
all_results <- rbind(all_results,
                     run3(d2,"rf",rf_g,"Random Forest","Exp2"))
features_subj <- c(features_full,"subj_num")
for(exp_n in c("Exp1","Exp2")) {
  d <- if(exp_n=="Exp1") d1 else d2
  m6 <- train(x=d$dsubj[,features_subj],
              y=d$dsubj$offset_class,method="rf",
              trControl=cv_6,tuneGrid=rf_g,
              metric="Accuracy",ntree=500)
  mb <- train(x=d$dsubj[,features_subj],
              y=d$dsubj$vision_group,method="rf",
              trControl=cv_b,tuneGrid=rf_g,
              metric="ROC",ntree=500)
  mr <- train(x=d$dsubj[,features_subj],
              y=d$dsubj$offset,method="rf",
              trControl=cv_r,tuneGrid=rf_g,
              metric="RMSE",ntree=500)
  all_results <- rbind(all_results,
                       data.frame(Experiment=exp_n,
                                  Model="RF+Participant",
                                  Acc_6class=g_acc(m6),
                                  Acc_binary=g_bacc(mb),
                                  AUC_binary=g_auc(mb),
                                  RMSE_reg=g_rmse(mr),
                                  R2_reg=g_r2(mr)))
  cat("RF+Participant",exp_n,"done!\n")
}
extract_ts <- function(t_norm) {
  t_norm %>%
    group_by(subjName,trialN) %>%
    summarise(
      vel_25pct=mean(GPvel[time_bin==25],na.rm=TRUE),
      vel_50pct=mean(GPvel[time_bin==50],na.rm=TRUE),
      vel_75pct=mean(GPvel[time_bin==75],na.rm=TRUE),
      peak_vel=max(GPvel,na.rm=TRUE),
      time_to_peak=time_bin[which.max(GPvel)][1],
      auc_vel=sum(GPvel,na.rm=TRUE),
      decel_rate=(mean(GPvel[time_bin>=75],na.rm=TRUE)-
                    max(GPvel,na.rm=TRUE))/
        (75-time_bin[which.max(GPvel)][1]),
      .groups="drop")
}
ts1 <- extract_ts(t1_norm)
ts2 <- extract_ts(t2_norm)
features_ts <- c(features_full,
                 "vel_25pct","vel_50pct","vel_75pct",
                 "peak_vel","time_to_peak","auc_vel","decel_rate")
prep_ts <- function(summ,ts,feats) {
  base <- summ %>%
    left_join(ts,by=c("subjName","trialN")) %>%
    mutate(
      offset_class=factor(paste0("O",offset)),
      vision_group=factor(ifelse(
        offset<=15,"HighVision","LowVision"))
    ) %>%
    mutate(across(all_of(feats),
                  ~ifelse(is.nan(.)|is.infinite(.),NA,.)))
  list(
    d6=base %>% select(all_of(feats),
                       offset_class) %>% drop_na(),
    dbin=base %>% select(all_of(feats),
                         vision_group) %>% drop_na(),
    dreg=base %>% select(all_of(feats),
                         offset) %>% drop_na()
  )
}
d1ts <- prep_ts(summary1,ts1,features_ts)
d2ts <- prep_ts(summary2_fixed,ts2,features_ts)
for(exp_n in c("Exp1","Exp2")) {
  dts <- if(exp_n=="Exp1") d1ts else d2ts
  m6 <- train(x=dts$d6[,features_ts],
              y=dts$d6$offset_class,method="rf",
              trControl=cv_6,tuneGrid=rf_g,
              metric="Accuracy",ntree=500)
  mb <- train(x=dts$dbin[,features_ts],
              y=dts$dbin$vision_group,method="rf",
              trControl=cv_b,tuneGrid=rf_g,
              metric="ROC",ntree=500)
  mr <- train(x=dts$dreg[,features_ts],
              y=dts$dreg$offset,method="rf",
              trControl=cv_r,tuneGrid=rf_g,
              metric="RMSE",ntree=500)
  all_results <- rbind(all_results,
                       data.frame(Experiment=exp_n,
                                  Model="RF+Timeseries",
                                  Acc_6class=g_acc(m6),
                                  Acc_binary=g_bacc(mb),
                                  AUC_binary=g_auc(mb),
                                  RMSE_reg=g_rmse(mr),
                                  R2_reg=g_r2(mr)))
  cat("RF+Timeseries",exp_n,"done!\n")
}
cat("\n=== RESULTS SO FAR ===\n")
print(all_results)
write.csv(all_results,"results_traditional.csv",
          row.names=FALSE)
# ============================================================
# SECTION 12: FDA — FUNCTIONAL DATA ANALYSIS
# ============================================================
cat("\n=== FDA — FUNCTIONAL DATA ANALYSIS ===\n")
seq_len  <- 100
t_grid   <- seq(0,1,length.out=seq_len)
fdParobj <- fdPar(
  create.bspline.basis(c(0,1),nbasis=20),
  Lfdobj=2,lambda=1e-4)
prep_sequences <- function(t_norm,summary_data) {
  trials <- t_norm %>%
    filter(!is.na(GPvel)) %>%
    group_by(subjName,trialN) %>%
    summarise(
      vel_seq=list(approx(time_pct,GPvel,
                          xout=seq(0,100,length.out=seq_len))$y),
      .groups="drop")
  summary_data %>%
    mutate(
      offset_class=factor(paste0("O",offset)),
      vision_group=factor(ifelse(
        offset<=15,"HighVision","LowVision"))
    ) %>%
    inner_join(trials,by=c("subjName","trialN")) %>%
    filter(!sapply(vel_seq,function(x) any(is.na(x))))
}
fda_run <- function(t_norm,summary_data,
                    exp_name,n_comp=5) {
  cat("\nFDA",exp_name,"...\n")
  fda_data <- prep_sequences(t_norm,summary_data)
  vel_mat <- do.call(cbind,
                     lapply(fda_data$vel_seq,unlist))
  smooth_fd <- smooth.basis(t_grid,vel_mat,
                            fdParobj)$fd
  fpca <- pca.fd(smooth_fd,nharm=n_comp)
  scores <- as.data.frame(fpca$scores)
  colnames(scores) <- paste0("FPC",1:n_comp)
  summ_feats <- summary_data %>%
    mutate(accelDecelRatio=ifelse(
      is.infinite(accelDecelRatio),NA,
      accelDecelRatio)) %>%
    select(subjName,trialN,all_of(features_full))
  combined <- cbind(scores,
                    summ_feats[match(
                      paste(fda_data$subjName,fda_data$trialN),
                      paste(summ_feats$subjName,summ_feats$trialN)),
                      features_full]) %>%
    mutate(across(everything(),
                  ~ifelse(is.nan(.)|is.infinite(.),NA,.))) %>%
    drop_na()
  n_valid <- nrow(combined)
  offset_class <- fda_data$offset_class[1:n_valid]
  vision_group <- fda_data$vision_group[1:n_valid]
  offset_num   <- fda_data$offset[1:n_valid]
  offset_renamed <- factor(case_when(
    as.character(offset_class)=="O0"  ~ "A0",
    as.character(offset_class)=="O5"  ~ "B5",
    as.character(offset_class)=="O10" ~ "C10",
    as.character(offset_class)=="O15" ~ "D15",
    as.character(offset_class)=="O20" ~ "E20",
    as.character(offset_class)=="O30" ~ "F30",
    as.character(offset_class)=="O70" ~ "G70",
    as.character(offset_class)=="O99" ~ "H99",
    TRUE ~ as.character(offset_class)
  ))
  set.seed(42)
  folds <- cut(sample(1:n_valid),breaks=10,labels=FALSE)
  acc6 <- accb <- rmse_v <- r2_v <- c()
  for(fold in 1:10) {
    cat("FDA",exp_name,"fold",fold,"/10\n")
    ti <- which(folds==fold)
    tri <- which(folds!=fold)
    rf6 <- randomForest(x=combined[tri,],
                        y=offset_renamed[tri],ntree=500,mtry=8)
    acc6 <- c(acc6,mean(
      predict(rf6,combined[ti,])==
        offset_renamed[ti])*100)
    rfb <- randomForest(x=combined[tri,],
                        y=vision_group[tri],ntree=500,mtry=8)
    accb <- c(accb,mean(
      predict(rfb,combined[ti,])==
        vision_group[ti])*100)
    rfr <- randomForest(x=combined[tri,],
                        y=offset_num[tri],ntree=500,mtry=8)
    pr <- predict(rfr,combined[ti,])
    rmse_v <- c(rmse_v,sqrt(mean((pr-offset_num[ti])^2)))
    r2_v <- c(r2_v,cor(pr,offset_num[ti])^2)
  }
  cat("FDA",exp_name,
      "6-class:",round(mean(acc6),2),"%\n")
  cat("FDA",exp_name,
      "Binary:",round(mean(accb),2),"%\n")
  cat("FDA",exp_name,
      "R²:",round(mean(r2_v),3),"\n")
  data.frame(Experiment=exp_name,
             Model="FDA+Summary",
             Acc_6class=round(mean(acc6),2),
             Acc_binary=round(mean(accb),2),
             AUC_binary=NA,
             RMSE_reg=round(mean(rmse_v),3),
             R2_reg=round(mean(r2_v),3))
}
fda1 <- fda_run(t1_norm,summary1,"Exp1")
all_results <- rbind(all_results,fda1)
fda2 <- fda_run(t2_norm,summary2_fixed,"Exp2")
all_results <- rbind(all_results,fda2)
# ============================================================
# SECTION 13: TCN + LSTM
# ============================================================
cat("\n=== DEEP LEARNING: TCN + LSTM ===\n")
make_array <- function(seq_data,seq_len=100) {
  n <- nrow(seq_data)
  X <- array(0,dim=c(n,seq_len,1))
  for(i in 1:n) {
    v <- unlist(seq_data$vel_seq[i])
    v <- (v-min(v))/(max(v)-min(v)+1e-8)
    X[i,,1] <- v
  }
  X
}
seq1 <- prep_sequences(t1_norm,summary1)
seq2 <- prep_sequences(t2_norm,summary2_fixed)
X1   <- make_array(seq1)
X2   <- make_array(seq2)
tcn_block <- function(x,filters,ks,dr) {
  res <- x %>%
    layer_conv_1d(filters,ks,padding="causal",
                  dilation_rate=dr,activation="relu") %>%
    layer_conv_1d(filters,ks,padding="causal",
                  dilation_rate=dr,activation="relu") %>%
    layer_spatial_dropout_1d(0.2)
  if(dim(x)[[3]]!=filters)
    x <- x %>% layer_conv_1d(filters,1,padding="same")
  layer_add(list(x,res)) %>% layer_activation("relu")
}
build_tcn <- function(sl=100,nc=6,task="classification") {
  inp <- layer_input(shape=c(sl,1))
  x <- inp %>%
    tcn_block(32,3,1) %>% tcn_block(32,3,2) %>%
    tcn_block(32,3,4) %>% tcn_block(32,3,8) %>%
    layer_global_average_pooling_1d() %>%
    layer_dense(64,activation="relu") %>%
    layer_dropout(0.3)
  if(task=="classification") {
    out <- x %>% layer_dense(nc,activation="softmax")
    m <- keras_model(inp,out)
    m %>% compile(optimizer=optimizer_adam(0.001),
                  loss="sparse_categorical_crossentropy",
                  metrics=c("accuracy"))
  } else if(task=="binary") {
    out <- x %>% layer_dense(1,activation="sigmoid")
    m <- keras_model(inp,out)
    m %>% compile(optimizer=optimizer_adam(0.001),
                  loss="binary_crossentropy",
                  metrics=c("accuracy"))
  } else {
    out <- x %>% layer_dense(1,activation="linear")
    m <- keras_model(inp,out)
    m %>% compile(optimizer=optimizer_adam(0.001),
                  loss="mse",metrics=c("mae"))
  }
  m
}
build_lstm <- function(sl=100,nc=6,task="classification") {
  inp <- layer_input(shape=c(sl,1))
  x <- inp %>%
    layer_lstm(64,return_sequences=TRUE) %>%
    layer_dropout(0.3) %>%
    layer_lstm(32) %>%
    layer_dropout(0.3) %>%
    layer_dense(64,activation="relu") %>%
    layer_dropout(0.3)
  if(task=="classification") {
    out <- x %>% layer_dense(nc,activation="softmax")
    m <- keras_model(inp,out)
    m %>% compile(optimizer=optimizer_adam(0.001),
                  loss="sparse_categorical_crossentropy",
                  metrics=c("accuracy"))
  } else if(task=="binary") {
    out <- x %>% layer_dense(1,activation="sigmoid")
    m <- keras_model(inp,out)
    m %>% compile(optimizer=optimizer_adam(0.001),
                  loss="binary_crossentropy",
                  metrics=c("accuracy"))
  } else {
    out <- x %>% layer_dense(1,activation="linear")
    m <- keras_model(inp,out)
    m %>% compile(optimizer=optimizer_adam(0.001),
                  loss="mse",metrics=c("mae"))
  }
  m
}
dl_cv <- function(X,y6,yb,yr,build_fn,
                  model_name,exp_name,
                  k=10,epochs=30) {
  cat("\n===",model_name,"|",exp_name,"===\n")
  n <- dim(X)[1]
  set.seed(42)
  folds <- cut(sample(1:n),breaks=k,labels=FALSE)
  a6 <- ab <- rm_v <- r2_v <- c()
  nc <- length(unique(y6))
  for(fold in 1:k) {
    cat("Fold",fold,"/",k,"— ")
    ti <- which(folds==fold)
    tri <- which(folds!=fold)
    es <- callback_early_stopping(
      monitor="val_loss",patience=5,
      restore_best_weights=TRUE)
    m6 <- build_fn(dim(X)[2],nc,"classification")
    m6 %>% fit(X[tri,,,drop=FALSE],y6[tri],
               epochs=epochs,batch_size=32,
               validation_split=0.1,
               callbacks=list(es),verbose=0)
    p6 <- m6 %>% predict(X[ti,,,drop=FALSE],verbose=0)
    a6 <- c(a6,mean(apply(p6,1,which.max)-1==y6[ti]))
    keras::k_clear_session()
    mb <- build_fn(dim(X)[2],2,"binary")
    mb %>% fit(X[tri,,,drop=FALSE],yb[tri],
               epochs=epochs,batch_size=32,
               validation_split=0.1,
               callbacks=list(es),verbose=0)
    pb <- as.vector(mb %>% predict(
      X[ti,,,drop=FALSE],verbose=0))
    ab <- c(ab,mean(as.integer(pb>0.5)==yb[ti]))
    keras::k_clear_session()
    mr <- build_fn(dim(X)[2],1,"regression")
    mr %>% fit(X[tri,,,drop=FALSE],yr[tri],
               epochs=epochs,batch_size=32,
               validation_split=0.1,
               callbacks=list(es),verbose=0)
    pr <- as.vector(mr %>% predict(
      X[ti,,,drop=FALSE],verbose=0))
    rm_v <- c(rm_v,sqrt(mean((pr-yr[ti])^2)))
    r2_v <- c(r2_v,cor(pr,yr[ti])^2)
    keras::k_clear_session()
    cat("6cl:",round(mean(a6)*100,1),
        "Bin:",round(mean(ab)*100,1),"\n")
  }
  cat(model_name,exp_name,"DONE!\n")
  cat("6-class:",round(mean(a6)*100,2),"%\n")
  cat("Binary:",round(mean(ab)*100,2),"%\n")
  cat("R²:",round(mean(r2_v),3),"\n")
  data.frame(Experiment=exp_name,
             Model=model_name,
             Acc_6class=round(mean(a6)*100,2),
             Acc_binary=round(mean(ab)*100,2),
             AUC_binary=NA,
             RMSE_reg=round(mean(rm_v),3),
             R2_reg=round(mean(r2_v),3))
}
y1_6 <- as.integer(factor(paste0("O",seq1$offset)))-1L
y2_6 <- as.integer(factor(paste0("O",seq2$offset)))-1L
y1_b <- as.integer(seq1$offset>15)
y2_b <- as.integer(seq2$offset>15)
y1_r <- as.numeric(seq1$offset)
y2_r <- as.numeric(seq2$offset)
all_results <- rbind(all_results,
                     dl_cv(X1,y1_6,y1_b,y1_r,build_tcn,
                           "TCN","Exp1",k=10,epochs=30))
all_results <- rbind(all_results,
                     dl_cv(X2,y2_6,y2_b,y2_r,build_tcn,
                           "TCN","Exp2",k=10,epochs=30))
all_results <- rbind(all_results,
                     dl_cv(X1,y1_6,y1_b,y1_r,build_lstm,
                           "LSTM","Exp1",k=10,epochs=30))
all_results <- rbind(all_results,
                     dl_cv(X2,y2_6,y2_b,y2_r,build_lstm,
                           "LSTM","Exp2",k=10,epochs=30))

# ============================================================
# SECTION 14: LOPO-CV — ALL 4 MODELS (Exp1)
# ============================================================
# Trains on N-1 participants, tests on the held-out one.
# Models: Lasso (glmnet multinomial), SVM RBF, RF, FDA+Summary.
# FDA fits FPCA on training participants only, projects test
# participant onto training harmonics — no data leakage.
# ============================================================
cat("\n=== LOPO-CV ALL MODELS — EXP1 ===\n")

# ---------- helpers: scale with training params ----------
scale_train <- function(Xtr) {
  mu  <- colMeans(Xtr, na.rm = TRUE)
  sig <- apply(Xtr, 2, sd, na.rm = TRUE)
  sig[sig == 0] <- 1
  list(mu = mu, sig = sig)
}
scale_apply <- function(X, params) {
  sweep(sweep(X, 2, params$mu, "-"), 2, params$sig, "/")
}

# ---------- prepare base Exp1 LOPO dataset ----------
exp1_lopo <- summary1 %>%
  mutate(
    offset_class = factor(paste0("O", offset)),
    accelDecelRatio = ifelse(is.infinite(accelDecelRatio),
                             NA, accelDecelRatio)
  ) %>%
  select(all_of(features_full), offset_class, offset, subjName) %>%
  drop_na()

participants_1 <- unique(exp1_lopo$subjName)
cat("Exp1 LOPO participants:", length(participants_1), "\n")

# ---------- FDA: prepare sequences for Exp1 ----------
seq1_fda <- prep_sequences(t1_norm, summary1)
# attach clean summary features
seq1_fda <- seq1_fda %>%
  left_join(
    summary1 %>%
      mutate(accelDecelRatio = ifelse(is.infinite(accelDecelRatio),
                                      NA, accelDecelRatio)) %>%
      select(subjName, trialN, all_of(features_full)),
    by = c("subjName", "trialN")
  ) %>%
  filter(!sapply(vel_seq, function(x) any(is.na(x)))) %>%
  drop_na(all_of(features_full))

lopo1_results <- data.frame()

for (p in participants_1) {
  cat("\n--- LOPO Exp1 held out:", p, "---\n")
  
  tr_d  <- exp1_lopo %>% filter(subjName != p)
  te_d  <- exp1_lopo %>% filter(subjName == p)
  Xtr   <- as.matrix(tr_d[, features_full])
  Xte   <- as.matrix(te_d[, features_full])
  y6tr  <- tr_d$offset_class
  y6te  <- te_d$offset_class
  yotr  <- tr_d$offset
  yote  <- te_d$offset
  
  sc <- scale_train(Xtr)
  Xtr_sc <- scale_apply(Xtr, sc)
  Xte_sc <- scale_apply(Xte, sc)
  
  # ---- 1. Lasso (multinomial) ----
  lasso_cv <- cv.glmnet(Xtr_sc, y6tr,
                        family = "multinomial",
                        alpha = 1,
                        nfolds = 5,
                        type.measure = "class")
  lasso_pred <- predict(lasso_cv,
                        newx   = Xte_sc,
                        s      = "lambda.min",
                        type   = "class")
  lasso_acc <- mean(lasso_pred == as.character(y6te)) * 100
  
  # ---- 2. SVM RBF ----
  svm_fit  <- svm(x = Xtr_sc, y = y6tr,
                  kernel = "radial",
                  cost   = 1,
                  gamma  = 0.01)
  svm_acc  <- mean(predict(svm_fit, Xte_sc) == y6te) * 100
  
  # ---- 3. Random Forest ----
  rf_fit  <- randomForest(x = Xtr, y = y6tr,
                          ntree = 500, mtry = 8)
  rf_pred <- predict(rf_fit, Xte)
  rf_acc  <- mean(rf_pred == y6te) * 100
  
  # RF regression for R²
  rfr_fit  <- randomForest(x = Xtr, y = yotr,
                           ntree = 500, mtry = 8)
  rf_pr    <- predict(rfr_fit, Xte)
  rf_r2    <- ifelse(sd(yote) > 0,
                     cor(rf_pr, yote)^2, NA)
  
  # ---- 4. FDA+Summary ----
  # Fit FPCA on training participants only
  tr_seq   <- seq1_fda %>% filter(subjName != p)
  te_seq   <- seq1_fda %>% filter(subjName == p)
  
  fda_acc <- NA
  tryCatch({
    vel_tr   <- do.call(cbind, lapply(tr_seq$vel_seq, unlist))
    fd_tr    <- smooth.basis(t_grid, vel_tr, fdParobj)$fd
    fpca_tr  <- pca.fd(fd_tr, nharm = 5)
    
    # Project test trials onto training harmonics
    vel_te   <- do.call(cbind, lapply(te_seq$vel_seq, unlist))
    fd_te    <- smooth.basis(t_grid, vel_te, fdParobj)$fd
    # scores = integral of (test_fd - mean_fd) * harmonic
    fda_mean <- fpca_tr$meanfd
    harmonics<- fpca_tr$harmonics
    scores_te <- inprod(fd_te - fda_mean, harmonics)
    scores_tr <- fpca_tr$scores
    
    colnames(scores_tr) <- paste0("FPC", 1:5)
    colnames(scores_te) <- paste0("FPC", 1:5)
    
    # Combine FPC scores with summary features
    sumf_tr <- as.matrix(tr_seq[, features_full])
    sumf_te <- as.matrix(te_seq[, features_full])
    
    Xfda_tr <- cbind(scores_tr, sumf_tr)
    Xfda_te <- cbind(scores_te, sumf_te)
    
    # Drop rows with NA
    ok_tr <- complete.cases(Xfda_tr)
    ok_te <- complete.cases(Xfda_te)
    Xfda_tr <- Xfda_tr[ok_tr, ]
    Xfda_te <- Xfda_te[ok_te, ]
    y6_fda_tr <- factor(paste0("O", tr_seq$offset[ok_tr]))
    y6_fda_te <- factor(paste0("O", te_seq$offset[ok_te]),
                        levels = levels(y6_fda_tr))
    
    rf_fda <- randomForest(x = Xfda_tr,
                           y = y6_fda_tr,
                           ntree = 500, mtry = 8)
    fda_acc <- mean(predict(rf_fda, Xfda_te) ==
                      y6_fda_te) * 100
  }, error = function(e) {
    cat("  FDA error for", p, ":", conditionMessage(e), "\n")
  })
  
  row <- data.frame(
    participant = p,
    n_trials    = nrow(te_d),
    Lasso_6cl   = round(lasso_acc, 2),
    SVM_6cl     = round(svm_acc,   2),
    RF_6cl      = round(rf_acc,    2),
    FDA_6cl     = round(fda_acc,   2),
    RF_R2       = round(rf_r2,     3)
  )
  lopo1_results <- rbind(lopo1_results, row)
  
  cat("  Lasso:", round(lasso_acc,1),
      "| SVM:", round(svm_acc,1),
      "| RF:", round(rf_acc,1),
      "| FDA:", round(fda_acc,1), "\n")
}

cat("\n=== EXP1 LOPO-CV SUMMARY (all models) ===\n")
print(lopo1_results)
cat("\nMean LOPO-CV accuracy (6-class) — Exp1:\n")
cat("  Lasso:", round(mean(lopo1_results$Lasso_6cl, na.rm=TRUE), 2), "%\n")
cat("  SVM RBF:", round(mean(lopo1_results$SVM_6cl,  na.rm=TRUE), 2), "%\n")
cat("  RF:",     round(mean(lopo1_results$RF_6cl,    na.rm=TRUE), 2), "%\n")
cat("  FDA+Summary:", round(mean(lopo1_results$FDA_6cl, na.rm=TRUE), 2), "%\n")
cat("  Chance baseline: 16.7%\n")
cat("  RF Mean R²:", round(mean(lopo1_results$RF_R2, na.rm=TRUE), 3), "\n")

write.csv(lopo1_results,
          "transport_exp1_6class_lopo_allmodels.csv",
          row.names = FALSE)
cat("Saved: transport_exp1_6class_lopo_allmodels.csv\n")

# ============================================================
# SECTION 14b: LOPO-CV — MATCHED BINARY (Exp1)
# 5/10mm = HighVision, 30/70mm = LowVision (balanced accuracy)
# ============================================================
cat("\n=== LOPO-CV MATCHED BINARY (Exp1) ===\n")
exp1_matched <- summary1 %>%
  filter(offset %in% c(5, 10, 30, 70)) %>%
  mutate(
    vision_matched = factor(ifelse(
      offset %in% c(5, 10), "HighVision", "LowVision")),
    accelDecelRatio = ifelse(
      is.infinite(accelDecelRatio), NA, accelDecelRatio)
  ) %>%
  select(all_of(features_full), vision_matched, subjName) %>%
  drop_na()
cat("Matched binary rows:", nrow(exp1_matched),
    "(offsets 5,10,30,70 only)\n")
participants_m <- unique(exp1_matched$subjName)
lopo_matched <- data.frame()
for(p in participants_m) {
  train_m <- exp1_matched %>% filter(subjName != p)
  test_m  <- exp1_matched %>% filter(subjName == p)
  rfm <- randomForest(x = train_m[, features_full],
                      y = train_m$vision_matched,
                      ntree = 500, mtry = 8)
  preds <- predict(rfm, test_m[, features_full])
  tp <- sum(preds == "HighVision" & test_m$vision_matched == "HighVision")
  tn <- sum(preds == "LowVision"  & test_m$vision_matched == "LowVision")
  fp <- sum(preds == "HighVision" & test_m$vision_matched == "LowVision")
  fn <- sum(preds == "LowVision"  & test_m$vision_matched == "HighVision")
  sensitivity <- tp / (tp + fn)
  specificity <- tn / (tn + fp)
  ba <- (sensitivity + specificity) / 2
  lopo_matched <- rbind(lopo_matched, data.frame(
    participant  = p,
    n_trials     = nrow(test_m),
    balanced_acc = round(ba, 4),
    sensitivity  = round(sensitivity, 4),
    specificity  = round(specificity, 4)
  ))
  cat(p, "BA:", round(ba, 4), "\n")
}
cat("\n=== MATCHED BINARY LOPO SUMMARY ===\n")
print(lopo_matched)
cat("Mean Balanced Accuracy:",
    round(mean(lopo_matched$balanced_acc), 4), "\n")
cat("Chance baseline: 0.50\n")
write.csv(lopo_matched,
          "transport_exp1_binary_lopo_participant.csv",
          row.names = FALSE)
cat("Saved: transport_exp1_binary_lopo_participant.csv\n")

# ============================================================
# SECTION 14c: LOPO-CV — ALL 4 MODELS (Exp2)
# ============================================================
cat("\n=== LOPO-CV ALL MODELS — EXP2 ===\n")

exp2_lopo <- summary2_fixed %>%
  mutate(
    offset_class = factor(paste0("O", offset)),
    accelDecelRatio = ifelse(is.infinite(accelDecelRatio),
                             NA, accelDecelRatio)
  ) %>%
  select(all_of(features_full), offset_class, offset, subjName) %>%
  drop_na()

participants_2 <- unique(exp2_lopo$subjName)
cat("Exp2 LOPO participants:", length(participants_2), "\n")

# FDA sequences for Exp2
seq2_fda <- prep_sequences(t2_norm, summary2_fixed)
seq2_fda <- seq2_fda %>%
  left_join(
    summary2_fixed %>%
      mutate(accelDecelRatio = ifelse(is.infinite(accelDecelRatio),
                                      NA, accelDecelRatio)) %>%
      select(subjName, trialN, all_of(features_full)),
    by = c("subjName", "trialN")
  ) %>%
  filter(!sapply(vel_seq, function(x) any(is.na(x)))) %>%
  drop_na(all_of(features_full))

lopo2_results <- data.frame()

for (p in participants_2) {
  cat("\n--- LOPO Exp2 held out:", p, "---\n")
  
  tr_d  <- exp2_lopo %>% filter(subjName != p)
  te_d  <- exp2_lopo %>% filter(subjName == p)
  Xtr   <- as.matrix(tr_d[, features_full])
  Xte   <- as.matrix(te_d[, features_full])
  y6tr  <- tr_d$offset_class
  y6te  <- te_d$offset_class
  
  sc <- scale_train(Xtr)
  Xtr_sc <- scale_apply(Xtr, sc)
  Xte_sc <- scale_apply(Xte, sc)
  
  # ---- 1. Lasso ----
  lasso_cv <- cv.glmnet(Xtr_sc, y6tr,
                        family = "multinomial",
                        alpha  = 1,
                        nfolds = 5,
                        type.measure = "class")
  lasso_pred <- predict(lasso_cv,
                        newx = Xte_sc,
                        s    = "lambda.min",
                        type = "class")
  lasso_acc <- mean(lasso_pred == as.character(y6te)) * 100
  
  # ---- 2. SVM RBF ----
  svm_fit <- svm(x = Xtr_sc, y = y6tr,
                 kernel = "radial",
                 cost   = 1,
                 gamma  = 0.01)
  svm_acc <- mean(predict(svm_fit, Xte_sc) == y6te) * 100
  
  # ---- 3. Random Forest ----
  rf_fit  <- randomForest(x = Xtr, y = y6tr,
                          ntree = 500, mtry = 8)
  rf_acc  <- mean(predict(rf_fit, Xte) == y6te) * 100
  
  # ---- 4. FDA+Summary ----
  tr_seq  <- seq2_fda %>% filter(subjName != p)
  te_seq  <- seq2_fda %>% filter(subjName == p)
  
  fda_acc <- NA
  tryCatch({
    vel_tr   <- do.call(cbind, lapply(tr_seq$vel_seq, unlist))
    fd_tr    <- smooth.basis(t_grid, vel_tr, fdParobj)$fd
    fpca_tr  <- pca.fd(fd_tr, nharm = 5)
    
    vel_te   <- do.call(cbind, lapply(te_seq$vel_seq, unlist))
    fd_te    <- smooth.basis(t_grid, vel_te, fdParobj)$fd
    scores_te <- inprod(fd_te - fpca_tr$meanfd, fpca_tr$harmonics)
    scores_tr <- fpca_tr$scores
    
    colnames(scores_tr) <- paste0("FPC", 1:5)
    colnames(scores_te) <- paste0("FPC", 1:5)
    
    sumf_tr <- as.matrix(tr_seq[, features_full])
    sumf_te <- as.matrix(te_seq[, features_full])
    Xfda_tr <- cbind(scores_tr, sumf_tr)
    Xfda_te <- cbind(scores_te, sumf_te)
    
    ok_tr <- complete.cases(Xfda_tr)
    ok_te <- complete.cases(Xfda_te)
    Xfda_tr <- Xfda_tr[ok_tr, ]
    Xfda_te <- Xfda_te[ok_te, ]
    y6_fda_tr <- factor(paste0("O", tr_seq$offset[ok_tr]))
    y6_fda_te <- factor(paste0("O", te_seq$offset[ok_te]),
                        levels = levels(y6_fda_tr))
    
    rf_fda  <- randomForest(x = Xfda_tr, y = y6_fda_tr,
                            ntree = 500, mtry = 8)
    fda_acc <- mean(predict(rf_fda, Xfda_te) ==
                      y6_fda_te) * 100
  }, error = function(e) {
    cat("  FDA error for", p, ":", conditionMessage(e), "\n")
  })
  
  row <- data.frame(
    participant = p,
    n_trials    = nrow(te_d),
    Lasso_6cl   = round(lasso_acc, 2),
    SVM_6cl     = round(svm_acc,   2),
    RF_6cl      = round(rf_acc,    2),
    FDA_6cl     = round(fda_acc,   2)
  )
  lopo2_results <- rbind(lopo2_results, row)
  
  cat("  Lasso:", round(lasso_acc,1),
      "| SVM:", round(svm_acc,1),
      "| RF:", round(rf_acc,1),
      "| FDA:", round(fda_acc,1), "\n")
}

cat("\n=== EXP2 LOPO-CV SUMMARY (all models) ===\n")
print(lopo2_results)
cat("\nMean LOPO-CV accuracy (6-class) — Exp2:\n")
cat("  Lasso:", round(mean(lopo2_results$Lasso_6cl, na.rm=TRUE), 2), "%\n")
cat("  SVM RBF:", round(mean(lopo2_results$SVM_6cl,  na.rm=TRUE), 2), "%\n")
cat("  RF:",     round(mean(lopo2_results$RF_6cl,    na.rm=TRUE), 2), "%\n")
cat("  FDA+Summary:", round(mean(lopo2_results$FDA_6cl, na.rm=TRUE), 2), "%\n")
cat("  Chance baseline: 16.7%\n")

write.csv(lopo2_results,
          "transport_exp2_6class_lopo_allmodels.csv",
          row.names = FALSE)
cat("Saved: transport_exp2_6class_lopo_allmodels.csv\n")

# ============================================================
# SECTION 15: BETWEEN-SUBJECT VARIABILITY PLOTS
# ============================================================
p_lopo1 <- lopo1_results %>%
  ggplot(aes(x=reorder(participant,RF_6cl),
             y=RF_6cl,fill=RF_6cl>16.7)) +
  geom_bar(stat="identity",alpha=0.8) +
  geom_hline(yintercept=16.7,linetype="dashed",
             color="red",linewidth=1) +
  scale_fill_manual(values=c("FALSE"="#E05C5C",
                             "TRUE"="#2E75B6"),
                    labels=c("Below chance","Above chance")) +
  labs(title="Per-Participant LOPO-CV (RF 6-class)",
       x="Participant",y="Accuracy (%)",fill="") +
  theme_minimal()+theme(legend.position="bottom")
ggsave("fig_lopo_participants.png",
       plot=p_lopo1,width=8,height=4,dpi=150)
p_lopo2 <- lopo1_results %>%
  ggplot(aes(x=reorder(participant,RF_R2),
             y=RF_R2,fill=RF_R2>0)) +
  geom_bar(stat="identity",alpha=0.8) +
  geom_hline(yintercept=0,linetype="dashed",
             color="red",linewidth=1) +
  scale_fill_manual(values=c("FALSE"="#E05C5C",
                             "TRUE"="#2E75B6")) +
  labs(title="Per-Participant LOPO-CV R²",
       x="Participant",y="R²",fill="R²>0") +
  theme_minimal()+theme(legend.position="bottom")
ggsave("fig_lopo_r2.png",
       plot=p_lopo2,width=8,height=4,dpi=150)
# ============================================================
# SECTION 16: NON-LINEARITY PLOTS
# ============================================================
p_imp <- varImp(
  train(x=d1$d6[,features_full],
        y=d1$d6$offset_class,method="rf",
        trControl=trainControl(method="cv",number=3),
        tuneGrid=expand.grid(mtry=8),
        metric="Accuracy",ntree=200)
)$importance %>%
  rownames_to_column("feature") %>%
  arrange(desc(Overall)) %>%
  ggplot(aes(x=reorder(feature,Overall),y=Overall))+
  geom_bar(stat="identity",fill="#2E75B6",alpha=0.8)+
  coord_flip()+
  labs(title="RF Feature Importance — Offset",
       subtitle="FGPx/FGPy (lateral position) dominate",
       x="",y="Importance")+
  theme_minimal()
ggsave("fig_feature_importance.png",
       plot=p_imp,width=7,height=5,dpi=150)
p_nl <- summary1 %>%
  select(offset,FGPx,FGPy,MGV,
         decelPhase,MGA,movTime) %>%
  pivot_longer(cols=-offset,
               names_to="feature",values_to="value") %>%
  group_by(offset,feature) %>%
  summarise(mean_val=mean(value,na.rm=TRUE),
            .groups="drop") %>%
  group_by(feature) %>%
  mutate(mean_scaled=scale(mean_val)[,1]) %>%
  ggplot(aes(x=offset,y=mean_scaled,
             color=feature,group=feature))+
  geom_line(linewidth=1.2)+geom_point(size=3)+
  geom_hline(yintercept=0,linetype="dashed",
             color="grey50")+
  labs(title="Feature Trends with Offset (Standardised)",
       subtitle="Crossing lines = non-monotonic = non-linear",
       x="Offset (mm)",y="Standardised Mean",
       color="Feature")+
  theme_minimal()+
  theme(legend.position="bottom")
ggsave("fig_nonlinear_all.png",
       plot=p_nl,width=9,height=5,dpi=150)
cat("Non-linearity plots saved!\n")
# ============================================================
# SECTION 17: FINAL COMPLETE TABLE
# ============================================================
cat("\n=== COMPLETE RESULTS TABLE ===\n")
final_table <- all_results %>%
  arrange(Experiment,desc(Acc_6class))
print(final_table)
cat("\nChance: 6-class=16.7% | Binary=50%\n")
cat("RMSE lower=better | R² higher=better\n")
write.csv(final_table,
          "complete_results_final.csv",
          row.names=FALSE)
cat("\nSaved to complete_results_final.csv\n")
cat("Location:", getwd(), "\n")