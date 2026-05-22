suppressPackageStartupMessages({
  library(tidyverse)
  library(meta)
  library(data.table)
  library(gridExtra)
  library(ggrepel)
})

pdf(NULL)

settings.meta(
  col.square.lines = "transparent",
  col.diamond.lines = "transparent",
  text.tau2 = "τ²",
  text.I2 = "I²"
)

# --- Part 0: Configuration ---
CORRELATION_R <- 0.85
TIMESTAMP     <- format(Sys.time(), "%Y-%m-%d_%H-%M")
OUTPUT_BASE_DIR <- file.path(dirname(sys.frame(1)$ofile), "..", "output")
OUTPUT_FOLDER <- OUTPUT_BASE_DIR

PALETTE <- c(
  "Overall"     = "#3C5488",
  "AD"          = "#E64B35",
  "CN"          = "#4DBBD5",
  "MCI"         = "#E6A0C4",
  "Homo"        = "#B09C85",
  "Hetero"      = "#00A087",
  "Corrected"   = "#8491B4",
  "Uncorrected" = "#91D1C2",
  "Bias"        = "black"
)

THEME_MINIMAL <- theme_classic(base_size = 16) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 19, face = "bold"),
    axis.title = element_text(size = 17, face = "bold"),
    axis.text = element_text(size = 14),
    axis.line = element_line(linewidth = 0.8),
    axis.ticks = element_line(linewidth = 0.8),
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 14, face = "bold"),
    legend.background = element_blank(),
    legend.key = element_blank()
  )

file_path <- file.path(dirname(sys.frame(1)$ofile), "..", "data", "meta.csv")
if (!file.exists(file_path)) stop(paste("Data file not found:", file_path))
print(paste("Using input file:", file_path))
if (!dir.exists(OUTPUT_FOLDER)) dir.create(OUTPUT_FOLDER, recursive = TRUE)

# --- Helper Functions ---
fmt_p_exact <- function(p) {
  if (is.na(p)) return("NA")
  if (p < 0.0001) { return(formatC(p, format = "e", digits = 2)) }
  return(sprintf("%.4f", p))
}

extract_row <- function(model, name) {
  if (is.null(model)) return(NULL)
  data.frame(Analysis = name, k = model$k, SMD = round(model$TE.random, 4),
             CI_Lower = round(model$lower.random, 4), CI_Upper = round(model$upper.random, 4),
             P_Value_Display = fmt_p_exact(model$pval.random), P_Value_Raw = model$pval.random,
             I2 = paste0(round(model$I2 * 100, 1), "%"), Tau2 = round(model$tau2, 4))
}

extract_reg <- function(reg_model, name) {
  if (is.null(reg_model)) return(NULL)
  data.frame(Predictor = name, Coef = round(reg_model$b[2], 4), 
             SE = round(reg_model$se[2], 4), P_Value_Display = fmt_p_exact(reg_model$pval[2]),
             P_Value_Raw = reg_model$pval[2])
}

# --- Part 1: Data Loading ---
print("Step 1/5: Data preparation...")
df_raw <- fread(file_path)
core_cols <- c("apoe4_n", "apoe4_Mean", "apoe4_SD", "no_apoe4_n", "no_apoe4_Mean", "no_apoe4_SD", "age")
df_cleaned <- df_raw %>% select(where(~!all(is.na(.)))) %>% filter(if_all(all_of(core_cols), ~ !is.na(.)))
if("female_percentage" %in% colnames(df_cleaned)) df_cleaned$female_percentage <- as.numeric(df_cleaned$female_percentage)

df_bilateral <- df_cleaned %>% filter(hemisphere == "bilateral") %>% 
  rename(n.e = apoe4_n, mean.e = apoe4_Mean, sd.e = apoe4_SD, n.c = no_apoe4_n, mean.c = no_apoe4_Mean, sd.c = no_apoe4_SD)
df_combined <- df_cleaned %>% filter(hemisphere %in% c("left", "right")) %>% 
  pivot_wider(id_cols = any_of(c("title", "diagnosis", "apoe4_n", "no_apoe4_n", "age", "correction_method", "year", "e4_dosage", "female_percentage", "segmentation_method")),
              names_from = hemisphere, values_from = c(apoe4_Mean, apoe4_SD, no_apoe4_Mean, no_apoe4_SD)) %>% 
  mutate(mean.e = apoe4_Mean_left + apoe4_Mean_right,
         sd.e   = sqrt(apoe4_SD_left^2 + apoe4_SD_right^2 + 2 * CORRELATION_R * apoe4_SD_left * apoe4_SD_right),
         mean.c = no_apoe4_Mean_left + no_apoe4_Mean_right,
         sd.c   = sqrt(no_apoe4_SD_left^2 + no_apoe4_SD_right^2 + 2 * CORRELATION_R * no_apoe4_SD_right * no_apoe4_SD_right)) %>% 
  rename(n.e = apoe4_n, n.c = no_apoe4_n)
df_final <- bind_rows(df_bilateral, df_combined) %>% rename(studlab = title) %>%
  mutate(studlab = str_replace_all(studlab, "[^A-Za-z0-9 ]", "")) %>%
  select(studlab, diagnosis, n.e, mean.e, sd.e, n.c, mean.c, sd.c, age, e4_dosage, correction_method, female_percentage, segmentation_method)
write.csv(df_final, file.path(OUTPUT_FOLDER, "Data_Cleaned_Input.csv"), row.names = FALSE)

df_corrected <- df_final %>% filter(correction_method != "None")
df_uncorrected <- df_final %>% filter(correction_method == "None")
write.csv(df_corrected, file.path(OUTPUT_FOLDER, "Data_Corrected_Input.csv"), row.names = FALSE)
write.csv(df_uncorrected, file.path(OUTPUT_FOLDER, "Data_Uncorrected_Input.csv"), row.names = FALSE)

# --- Helper Functions for Plotting ---
draw_forest_clean <- function(meta_obj, filename, title, color_base) {
  if (is.null(meta_obj) || meta_obj$k < 2) return(NULL)
  png(filename=file.path(OUTPUT_FOLDER, filename), width=9, height=max(6, meta_obj$k*0.35+2.5), units="in", res=600, type="cairo")
  par(mar = c(1, 0, 1, 0), font.main = 2, cex.main = 1.3)
  forest(meta_obj, studlab=FALSE, layout="JAMA", comb.fixed=FALSE, header.line=TRUE, leftcols=c("n.e","mean.e","sd.e","n.c","mean.c","sd.c"), leftlabs=c("N","Mean","SD","N","Mean","SD"), rightcols=c("effect","ci","w.random"), rightlabs=c("SMD","95% CI","Weight"), label.e="APOE e4 (+)", label.c="APOE e4 (-)", col.square=color_base, col.diamond="maroon", col.inside="black", col.square.lines="transparent", col.diamond.lines="transparent", col.study=color_base, fontsize=12, digits=2, digits.pval=3, print.tau2=TRUE, print.I2=TRUE, print.pval.Q=TRUE, just.addcols="center", main=title)
  dev.off()
}

draw_bubble_plot <- function(reg_model, moderator_name, filename, color_point) {
  if (is.null(reg_model)) return(NULL)
  plot_data <- data.frame(y=reg_model$yi, x=reg_model$X[,2], w=1/reg_model$vi)
  intercept <- reg_model$b[1]; slope <- reg_model$b[2]; p_val <- reg_model$pval[2]
  p <- ggplot(plot_data, aes(x=x, y=y)) + geom_point(aes(size=w), color=color_point, alpha=0.6) + geom_abline(intercept=intercept, slope=slope, color="black", size=1, linetype="solid") + annotate("text", x=min(plot_data$x), y=max(plot_data$y), label=paste0("Slope P = ", fmt_p_exact(p_val)), hjust=0, vjust=1, size=6, fontface="bold") + scale_size_continuous(range=c(2,8), guide="none") + labs(title=paste("Meta-Regression:", moderator_name), x=moderator_name, y="Effect Size (SMD)") + THEME_MINIMAL
  ggsave(file.path(OUTPUT_FOLDER, filename), p, width=7, height=6, dpi=600)
}

# --- Main Analysis Function ---
run_full_analysis <- function(data, prefix) {
  if(nrow(data) < 2) { print(paste("Skipping", prefix, "- insufficient data")); return(NULL) }
  print(paste("Analyzing", prefix, "group..."))

  # Meta-analysis models
  meta_main <- metacont(data=data, n.e=n.e, mean.e=mean.e, sd.e=sd.e, n.c=n.c, mean.c=mean.c, sd.c=sd.c, studlab=studlab, sm="SMD", method.tau="REML", hakn=TRUE, comb.fixed=FALSE)
  meta_ad <- if(sum(data$diagnosis=="AD")>1) metacont(data=filter(data, diagnosis=="AD"), n.e=n.e, mean.e=mean.e, sd.e=sd.e, n.c=n.c, mean.c=mean.c, sd.c=sd.c, studlab=studlab, sm="SMD", method.tau="REML", hakn=TRUE, comb.fixed=FALSE) else NULL
  meta_cn <- if(sum(data$diagnosis=="CN")>1) metacont(data=filter(data, diagnosis=="CN"), n.e=n.e, mean.e=mean.e, sd.e=sd.e, n.c=n.c, mean.c=mean.c, sd.c=sd.c, studlab=studlab, sm="SMD", method.tau="REML", hakn=TRUE, comb.fixed=FALSE) else NULL
  meta_mci <- if(sum(data$diagnosis=="MCI")>1) metacont(data=filter(data, diagnosis=="MCI"), n.e=n.e, mean.e=mean.e, sd.e=sd.e, n.c=n.c, mean.c=mean.c, sd.c=sd.c, studlab=studlab, sm="SMD", method.tau="REML", hakn=TRUE, comb.fixed=FALSE) else NULL
  meta_homo <- if(sum(data$e4_dosage==2, na.rm=TRUE)>1) metacont(data=filter(data, e4_dosage==2), n.e=n.e, mean.e=mean.e, sd.e=sd.e, n.c=n.c, mean.c=mean.c, sd.c=sd.c, studlab=studlab, sm="SMD", comb.fixed=FALSE) else NULL
  meta_het <- if(sum(data$e4_dosage==1, na.rm=TRUE)>1) metacont(data=filter(data, e4_dosage==1), n.e=n.e, mean.e=mean.e, sd.e=sd.e, n.c=n.c, mean.c=mean.c, sd.c=sd.c, studlab=studlab, sm="SMD", comb.fixed=FALSE) else NULL

  tf_model <- trimfill(meta_main)
  loo_model <- metainf(meta_main, pooled="random")
  meta_reg_age <- metareg(meta_main, ~age)
  meta_reg_sex <- if(sum(!is.na(data$female_percentage))>=3) metareg(meta_main, ~female_percentage) else NULL
  meta_reg_cn_age <- if(!is.null(meta_cn) && meta_cn$k>=3) metareg(meta_cn, ~age) else NULL

  # Tables
  results_list <- list(extract_row(meta_main, "Overall"), extract_row(meta_ad, "Subgroup: AD"), extract_row(meta_mci, "Subgroup: MCI"), extract_row(meta_cn, "Subgroup: CN"), extract_row(tf_model, "Bias Corrected"))
  write.csv(bind_rows(results_list), file.path(OUTPUT_FOLDER, paste0("Table_", prefix, "_Main_Results_Summary.csv")), row.names=FALSE)
  
  dosage_list <- list(extract_row(meta_homo, "Dosage: Homozygotes"), extract_row(meta_het, "Dosage: Heterozygotes"))
  write.csv(bind_rows(dosage_list), file.path(OUTPUT_FOLDER, paste0("Table_", prefix, "_Dosage_Analysis.csv")), row.names=FALSE)
  
  reg_list <- list(extract_reg(meta_reg_age, "Age (Overall)"), extract_reg(meta_reg_sex, "Female % (Overall)"), extract_reg(meta_reg_cn_age, "Age (CN Subgroup)"))
  write.csv(bind_rows(reg_list), file.path(OUTPUT_FOLDER, paste0("Table_", prefix, "_MetaRegression.csv")), row.names=FALSE)
  
  loo_df <- data.frame(Omitted_Study=loo_model$studlab, New_SMD=round(loo_model$TE, 4), New_I2=round(loo_model$I2*100, 1))
  write.csv(loo_df, file.path(OUTPUT_FOLDER, paste0("Table_", prefix, "_Leave_One_Out_Data.csv")), row.names=FALSE)

  # --- Leave-One-Out Forest Plot ---
  loo_filename <- if(prefix == "Corrected") "FigS3a_Forest_Corrected_LeaveOneOut.png" else "FigS3b_Forest_Uncorrected_LeaveOneOut.png"
  png(filename = file.path(OUTPUT_FOLDER, loo_filename),
      width = 9, height = max(6, meta_main$k * 0.35 + 2.5), units = "in", res = 600, type = "cairo")
  par(mar = c(1, 0, 1, 0), font.main = 2, cex.main = 1.3)
  forest(loo_model, studlab = FALSE, layout = "JAMA", comb.fixed = FALSE,
         col.square = PALETTE[["Overall"]], col.diamond = "maroon",
         col.inside = "black",
         col.square.lines = "transparent", col.diamond.lines = "transparent",
         fontsize = 12, digits = 2, digits.pval = 3,
         header.line = TRUE,
         just.addcols = "center",
         main = paste0(prefix, ": Leave-One-Out"),
         col.study = PALETTE[["Overall"]])
  dev.off()

  # Forest plots
  main_forest_filename <- if(prefix == "Corrected") "Fig3a_Forest_Corrected_Overall.png" else "Fig3b_Forest_Uncorrected_Overall.png"
  png(filename=file.path(OUTPUT_FOLDER, main_forest_filename), width=9, height=max(6, meta_main$k*0.35+2.5), units="in", res=600, type="cairo")
  par(mar = c(1, 0, 1, 0), font.main = 2, cex.main = 1.3)
  forest(meta_main, byvar=data$diagnosis, studlab=FALSE, layout="JAMA", comb.fixed=FALSE, header.line=TRUE, leftcols=c("n.e","mean.e","sd.e","n.c","mean.c","sd.c"), leftlabs=c("N","Mean","SD","N","Mean","SD"), rightcols=c("effect","ci","w.random"), rightlabs=c("SMD","95% CI","Weight"), col.square=PALETTE[["Overall"]], col.diamond="maroon", col.inside="black", col.square.lines="transparent", col.diamond.lines="transparent", col.study=PALETTE[["Overall"]], just.addcols="center", print.tau2=TRUE, print.I2=TRUE, print.pval.Q=TRUE, fontsize=12, digits=2, digits.pval=3, main=paste0(prefix, ": APOE4 Effect by Diagnosis"))
  dev.off()
  
  # Subgroup/dosage filenames based on prefix
  if (prefix == "Corrected") {
    fn_ad <- "FigS1a_Forest_Corrected_AD.png"
    fn_cn <- "FigS1b_Forest_Corrected_CN.png"
    fn_mci <- "Supp_Forest_Corrected_MCI.png"
    fn_homo <- "FigS1c_Forest_Corrected_Homo.png"
    fn_hetero <- "FigS1d_Forest_Corrected_Hetero.png"
  } else {
    fn_ad <- "Supp_Forest_Uncorrected_AD.png"
    fn_cn <- "FigS2a_Forest_Uncorrected_CN.png"
    fn_mci <- "Supp_Forest_Uncorrected_MCI.png"
    fn_homo <- "FigS2b_Forest_Uncorrected_Homo.png"
    fn_hetero <- "FigS2c_Forest_Uncorrected_Hetero.png"
  }
  draw_forest_clean(meta_ad, fn_ad, paste0(prefix, ": AD Subgroup"), PALETTE[["AD"]])
  draw_forest_clean(meta_cn, fn_cn, paste0(prefix, ": CN Subgroup"), PALETTE[["CN"]])
  draw_forest_clean(meta_mci, fn_mci, paste0(prefix, ": MCI Subgroup"), PALETTE[["MCI"]])
  draw_forest_clean(meta_homo, fn_homo, paste0(prefix, ": Homozygotes"), PALETTE[["Homo"]])
  draw_forest_clean(meta_het, fn_hetero, paste0(prefix, ": Heterozygotes"), PALETTE[["Hetero"]])

  # --- Funnel Plot with Trim-and-Fill ---
  funnel_filename <- if(prefix == "Corrected") "FigS4a_Funnel_Corrected_TrimFill.png" else "FigS4b_Funnel_Uncorrected_TrimFill.png"
  png(filename = file.path(OUTPUT_FOLDER, funnel_filename),
      width = 7, height = 6, units = "in", res = 600, type = "cairo")
  par(mar = c(4, 4, 2, 1), font.main = 2, cex.main = 1.3,
      cex.lab = 1.2, font.lab = 2, cex.axis = 1.1)
  funnel(tf_model,
         xlab = "Standardized Mean Difference (Hedges' g)",
         studlab = FALSE,
         main = paste0(prefix, ": Funnel Plot"))
  dev.off()

  # Prepare summary data for combined plot
  summary_data <- data.frame(
    Group=c("Overall","AD","CN","MCI","Homo","Hetero","Bias Corrected"),
    Category=c("Main","Diagnosis","Diagnosis","Diagnosis","Dose","Dose","Bias"),
    SMD=c(meta_main$TE.random, if(!is.null(meta_ad)) meta_ad$TE.random else NA, if(!is.null(meta_cn)) meta_cn$TE.random else NA, if(!is.null(meta_mci)) meta_mci$TE.random else NA, if(!is.null(meta_homo)) meta_homo$TE.random else NA, if(!is.null(meta_het)) meta_het$TE.random else NA, tf_model$TE.random),
    Lower=c(meta_main$lower.random, if(!is.null(meta_ad)) meta_ad$lower.random else NA, if(!is.null(meta_cn)) meta_cn$lower.random else NA, if(!is.null(meta_mci)) meta_mci$lower.random else NA, if(!is.null(meta_homo)) meta_homo$lower.random else NA, if(!is.null(meta_het)) meta_het$lower.random else NA, tf_model$lower.random),
    Upper=c(meta_main$upper.random, if(!is.null(meta_ad)) meta_ad$upper.random else NA, if(!is.null(meta_cn)) meta_cn$upper.random else NA, if(!is.null(meta_mci)) meta_mci$upper.random else NA, if(!is.null(meta_homo)) meta_homo$upper.random else NA, if(!is.null(meta_het)) meta_het$upper.random else NA, tf_model$upper.random),
    P_val=c(meta_main$pval.random, if(!is.null(meta_ad)) meta_ad$pval.random else NA, if(!is.null(meta_cn)) meta_cn$pval.random else NA, if(!is.null(meta_mci)) meta_mci$pval.random else NA, if(!is.null(meta_homo)) meta_homo$pval.random else NA, if(!is.null(meta_het)) meta_het$pval.random else NA, tf_model$pval.random)
  ) %>% filter(!is.na(SMD))

  # Baujat data
  b_res <- baujat(meta_main, plot=FALSE)
  baujat_data <- data.frame(
    x = b_res$x, y = b_res$y,
    studlab = meta_main$studlab,
    id = seq_len(meta_main$k)
  )
  
  # Meta-regression data
  reg_age_data <- if(!is.null(meta_reg_age)) data.frame(y=meta_reg_age$yi, x=meta_reg_age$X[,2], w=1/meta_reg_age$vi, intercept=meta_reg_age$b[1], slope=meta_reg_age$b[2], p_val=meta_reg_age$pval[2]) else NULL
  reg_sex_data <- if(!is.null(meta_reg_sex)) data.frame(y=meta_reg_sex$yi, x=meta_reg_sex$X[,2], w=1/meta_reg_sex$vi, intercept=meta_reg_sex$b[1], slope=meta_reg_sex$b[2], p_val=meta_reg_sex$pval[2]) else NULL
  reg_cn_age_data <- if(!is.null(meta_reg_cn_age)) data.frame(y=meta_reg_cn_age$yi, x=meta_reg_cn_age$X[,2], w=1/meta_reg_cn_age$vi, intercept=meta_reg_cn_age$b[1], slope=meta_reg_cn_age$b[2], p_val=meta_reg_cn_age$pval[2]) else NULL

  # Multiverse data
  # --- Sensitivity: remove most influential studies (Baujat-based) ---
  run_sens <- function(exclude_count) {
    influence_df <- data.frame(
      studlab     = meta_main$studlab,
      het_contrib = b_res$x,
      influence   = b_res$y,
      stringsAsFactors = FALSE
    )
    influence_df$dist <- sqrt(influence_df$het_contrib^2 + influence_df$influence^2)
    influence_df <- influence_df %>% arrange(desc(dist))
    exclude_studies <- influence_df$studlab[1:exclude_count]
    subset_data <- data %>% filter(!studlab %in% exclude_studies)
    m <- metacont(data = subset_data, n.e = n.e, mean.e = mean.e, sd.e = sd.e,
                  n.c = n.c, mean.c = mean.c, sd.c = sd.c, studlab = studlab,
                  sm = "SMD", method.tau = "REML", hakn = TRUE, comb.fixed = FALSE)
    return(data.frame(
      Scenario = paste0("Remove Top ", exclude_count, " Influential"),
      SMD = m$TE.random, Lower = m$lower.random, Upper = m$upper.random
    ))
  }
  max_exclude <- meta_main$k - 2
  multiverse_data <- bind_rows(
    data.frame(Scenario = "Original", SMD = meta_main$TE.random,
               Lower = meta_main$lower.random, Upper = meta_main$upper.random),
    if (max_exclude >= 1) run_sens(1) else NULL,
    if (max_exclude >= 3) run_sens(3) else NULL,
    if (max_exclude >= 5) run_sens(5) else NULL
  )

  return(list(meta_main=meta_main, meta_ad=meta_ad, meta_cn=meta_cn, meta_mci=meta_mci, meta_homo=meta_homo, meta_het=meta_het, tf_model=tf_model, summary_data=summary_data, baujat_data=baujat_data, reg_age_data=reg_age_data, reg_sex_data=reg_sex_data, reg_cn_age_data=reg_cn_age_data, multiverse_data=multiverse_data))
}

# --- Part 2: Execute analyses for both groups ---
print("Step 2/5: Running Corrected group analysis...")
results_corrected <- run_full_analysis(df_corrected, "Corrected")

print("Step 3/5: Running Uncorrected group analysis...")
results_uncorrected <- run_full_analysis(df_uncorrected, "Uncorrected")

# --- Part 3: Combined visualization ---
print("Step 4/5: Creating combined comparison plots...")
if(!is.null(results_corrected) && !is.null(results_uncorrected)) {

  # 1. Combined Summary plot
  comp_summary <- bind_rows(
    results_corrected$summary_data %>% mutate(Method="Corrected"),
    results_uncorrected$summary_data %>% mutate(Method="Uncorrected")
  )
  comp_summary$Group <- factor(comp_summary$Group, levels=rev(unique(comp_summary$Group)))
  # Format: combined label below each point
  comp_summary$below_label <- paste0(
    sprintf("%.2f [%.2f, %.2f]", comp_summary$SMD, comp_summary$Lower, comp_summary$Upper),
    ",  p = ", formatC(comp_summary$P_val, format = "e", digits = 2)
  )
  # Manual y offset: Corrected above, Uncorrected below; label further below each
  comp_summary$y_num <- as.numeric(comp_summary$Group)
  comp_summary$y_pos <- ifelse(comp_summary$Method == "Corrected",
                               comp_summary$y_num + 0.35,
                               comp_summary$y_num - 0.35)
  x_label_pos <- max(comp_summary$Upper, na.rm = TRUE) + 0.05
  p1 <- ggplot(comp_summary, aes(color=Method)) +
    geom_vline(xintercept=0, linetype="dashed", color="gray50") +
    geom_errorbarh(aes(xmin=Lower, xmax=Upper, y=y_pos), height=0.2) +
    geom_point(aes(x=SMD, y=y_pos), size=4) +
    geom_text(aes(x=x_label_pos, y=y_pos, label=below_label),
              size=4, color="black", hjust=0, show.legend=FALSE) +
    scale_y_continuous(breaks=seq_len(nlevels(comp_summary$Group)),
                       labels=levels(comp_summary$Group),
                       expand=expansion(add=c(0.8, 0.8))) +
    scale_color_manual(values=c("Corrected"=PALETTE[["Corrected"]], "Uncorrected"=PALETTE[["Uncorrected"]])) +
    scale_x_continuous(breaks = seq(-1.5, 0.5, 0.5)) +
    coord_cartesian(clip = "off") +
    labs(title="Effect Size Summary: ICV-Corrected vs ICV-Uncorrected", x="Effect Size (SMD)", y=NULL) +
    THEME_MINIMAL +
    theme(legend.position = c(1.15, 1.03), legend.justification = c(1, 1),
          axis.text.y=element_text(),
          plot.margin = margin(5.5, 140, 5.5, 5.5))
  ggsave(file.path(OUTPUT_FOLDER, "Fig3c_Combined_Summary.png"), p1, width=12, height=8, dpi=600, device=png, type="cairo")

  # 2. Combined Baujat plot (no outlier labels)
  comp_baujat <- bind_rows(
    results_corrected$baujat_data %>% mutate(Method = "Corrected"),
    results_uncorrected$baujat_data %>% mutate(Method = "Uncorrected")
  )
  p2 <- ggplot(comp_baujat, aes(x = x, y = y, color = Method)) +
    geom_point(size = 3, alpha = 0.7) +
    scale_color_manual(values = c("Corrected" = PALETTE[["Corrected"]],
                                  "Uncorrected" = PALETTE[["Uncorrected"]])) +
    labs(title = "Baujat Heterogeneity Diagnostic Plot",
         x = "Contribution to Overall Heterogeneity (Q)",
         y = "Influence on Pooled Result") +
    THEME_MINIMAL +
    theme(legend.position = "right")
  ggsave(file.path(OUTPUT_FOLDER, "Fig3d_Combined_Baujat.png"), p2, width = 8, height = 6, dpi = 600, device = png, type = "cairo")

  # 3. Combined Age Regression
  if(!is.null(results_corrected$reg_age_data) && !is.null(results_uncorrected$reg_age_data)) {
    comp_age <- bind_rows(results_corrected$reg_age_data %>% mutate(Method="Corrected"), results_uncorrected$reg_age_data %>% mutate(Method="Uncorrected"))
    p3 <- ggplot(comp_age, aes(x=x, y=y, color=Method, size=w)) + geom_point(alpha=0.6) + geom_abline(data=comp_age %>% distinct(Method, intercept, slope), aes(intercept=intercept, slope=slope, color=Method), size=1) + scale_color_manual(values=c("Corrected"=PALETTE[["Corrected"]], "Uncorrected"=PALETTE[["Uncorrected"]])) + scale_size_continuous(range=c(2,8), guide="none") + labs(title="Meta-Regression: Effect Size vs Mean Age", x="Mean Age", y="Effect Size (SMD)") + THEME_MINIMAL
    ggsave(file.path(OUTPUT_FOLDER, "FigS5a_MetaReg_Age.png"), p3, width=8, height=5.5, dpi=600, device=png, type="cairo")
  }

  # Combined Sex (Female %) Regression
  if (!is.null(results_corrected$reg_sex_data) && !is.null(results_uncorrected$reg_sex_data)) {
    comp_sex <- bind_rows(
      results_corrected$reg_sex_data %>% mutate(Method = "Corrected"),
      results_uncorrected$reg_sex_data %>% mutate(Method = "Uncorrected")
    )
    p_sex <- ggplot(comp_sex, aes(x = x, y = y, color = Method, size = w)) +
      geom_point(alpha = 0.6) +
      geom_abline(data = comp_sex %>% distinct(Method, intercept, slope),
                  aes(intercept = intercept, slope = slope, color = Method), linewidth = 1) +
      scale_color_manual(values = c("Corrected" = PALETTE[["Corrected"]],
                                    "Uncorrected" = PALETTE[["Uncorrected"]])) +
      scale_size_continuous(range = c(2, 8), guide = "none") +
      labs(title = "Meta-Regression: Effect Size vs Female Percentage",
           x = "Female Percentage (%)", y = "Effect Size (SMD)") +
      THEME_MINIMAL
    ggsave(file.path(OUTPUT_FOLDER, "FigS5b_MetaReg_Sex.png"), p_sex, width = 8, height = 5.5, dpi = 600, device = png, type = "cairo")
  }

  # Combined CN-subgroup Age Regression
  if (!is.null(results_corrected$reg_cn_age_data) && !is.null(results_uncorrected$reg_cn_age_data)) {
    comp_cn_age <- bind_rows(
      results_corrected$reg_cn_age_data %>% mutate(Method = "Corrected"),
      results_uncorrected$reg_cn_age_data %>% mutate(Method = "Uncorrected")
    )
    p_cn <- ggplot(comp_cn_age, aes(x = x, y = y, color = Method, size = w)) +
      geom_point(alpha = 0.6) +
      geom_abline(data = comp_cn_age %>% distinct(Method, intercept, slope),
                  aes(intercept = intercept, slope = slope, color = Method), linewidth = 1) +
      scale_color_manual(values = c("Corrected" = PALETTE[["Corrected"]],
                                    "Uncorrected" = PALETTE[["Uncorrected"]])) +
      scale_size_continuous(range = c(2, 8), guide = "none") +
      labs(title = "Meta-Regression: Effect Size vs Mean Age (CN Subgroup)",
           x = "Mean Age", y = "Effect Size (SMD)") +
      THEME_MINIMAL
    ggsave(file.path(OUTPUT_FOLDER, "FigS5c_MetaReg_CN_Age.png"), p_cn, width = 8, height = 5.5, dpi = 600, device = png, type = "cairo")
  }

  # 4. Combined Multiverse
  comp_multi <- bind_rows(results_corrected$multiverse_data %>% mutate(Method="Corrected"), results_uncorrected$multiverse_data %>% mutate(Method="Uncorrected"))
  p4 <- ggplot(comp_multi, aes(x=SMD, y=Scenario, color=Method)) + geom_vline(xintercept=0, linetype="dashed", color="gray50") + geom_errorbarh(aes(xmin=Lower, xmax=Upper), height=0.3, position=position_dodge(width=0.6)) + geom_point(size=4, position=position_dodge(width=0.6)) + scale_color_manual(values=c("Corrected"=PALETTE[["Corrected"]], "Uncorrected"=PALETTE[["Uncorrected"]])) + labs(title="Multiverse Sensitivity Analysis", x="Effect Size (SMD)", y=NULL) + THEME_MINIMAL
  ggsave(file.path(OUTPUT_FOLDER, "Fig3e_Combined_Multiverse.png"), p4, width=9, height=5, dpi=600, device=png, type="cairo")

}

print(paste("Analysis complete! Results in:", OUTPUT_FOLDER))

# ===================================================================
# Part 4: Sensitivity Analyses
# ===================================================================

# --- 4a. Excluding Wang 2019 (ADNI overlap) ---
print("Step 4a: Sensitivity - Excluding Wang 2019...")

wang_pattern <- "Relationship.*Hippocampal.*Delayed"
wang_rows <- df_corrected %>% filter(grepl(wang_pattern, studlab, ignore.case = TRUE))
cat("Wang 2019 cohorts found:\n")
print(wang_rows %>% select(studlab, diagnosis, n.e, n.c, correction_method))

df_corrected_no_wang <- df_corrected %>% filter(!grepl(wang_pattern, studlab, ignore.case = TRUE))

cat(sprintf("\nCorrected stratum: k = %d cohorts\n", nrow(df_corrected)))
cat(sprintf("Corrected stratum (excl. Wang 2019): k = %d cohorts\n\n", nrow(df_corrected_no_wang)))

meta_full_wang <- metacont(data=df_corrected, n.e=n.e, mean.e=mean.e, sd.e=sd.e,
                           n.c=n.c, mean.c=mean.c, sd.c=sd.c, studlab=studlab,
                           sm="SMD", method.tau="REML", hakn=TRUE, comb.fixed=FALSE)

meta_excl_wang <- metacont(data=df_corrected_no_wang, n.e=n.e, mean.e=mean.e, sd.e=sd.e,
                           n.c=n.c, mean.c=mean.c, sd.c=sd.c, studlab=studlab,
                           sm="SMD", method.tau="REML", hakn=TRUE, comb.fixed=FALSE)

cat(sprintf("=== FULL Corrected Stratum (k=%d) ===\n", meta_full_wang$k))
cat(sprintf("  SMD = %.4f, 95%% CI [%.4f, %.4f]\n", meta_full_wang$TE.random, meta_full_wang$lower.random, meta_full_wang$upper.random))
cat(sprintf("  p = %.4f, I2 = %.1f%%\n\n", meta_full_wang$pval.random, meta_full_wang$I2*100))

cat(sprintf("=== EXCLUDING Wang 2019 (k=%d) ===\n", meta_excl_wang$k))
cat(sprintf("  SMD = %.4f, 95%% CI [%.4f, %.4f]\n", meta_excl_wang$TE.random, meta_excl_wang$lower.random, meta_excl_wang$upper.random))
cat(sprintf("  p = %.4f, I2 = %.1f%%\n\n", meta_excl_wang$pval.random, meta_excl_wang$I2*100))

cat("=== COMPARISON ===\n")
cat(sprintf("  Full:      SMD = %.3f [%.3f, %.3f], p = %.4f, I2 = %.1f%%\n",
            meta_full_wang$TE.random, meta_full_wang$lower.random, meta_full_wang$upper.random, meta_full_wang$pval.random, meta_full_wang$I2*100))
cat(sprintf("  Excl Wang: SMD = %.3f [%.3f, %.3f], p = %.4f, I2 = %.1f%%\n",
            meta_excl_wang$TE.random, meta_excl_wang$lower.random, meta_excl_wang$upper.random, meta_excl_wang$pval.random, meta_excl_wang$I2*100))
cat(sprintf("  Delta SMD = %.4f (%.1f%% change)\n",
            meta_excl_wang$TE.random - meta_full_wang$TE.random,
            abs(meta_excl_wang$TE.random - meta_full_wang$TE.random) / abs(meta_full_wang$TE.random) * 100))

sig_full <- ifelse(meta_full_wang$pval.random < 0.05, "Significant", "Non-significant")
sig_excl <- ifelse(meta_excl_wang$pval.random < 0.05, "Significant", "Non-significant")
cat(sprintf("  Significance: Full = %s, Excl = %s\n", sig_full, sig_excl))
cat(sprintf("  Conclusion change: %s\n\n", ifelse(sig_full == sig_excl, "NO - conclusion robust", "YES - conclusion changes!")))

# --- 4b. Subgroup by Segmentation Method ---
print("Step 4b: Sensitivity - Subgroup by Segmentation Method...")

cat("\n=== Segmentation Method Subgroup Analysis (ICV-Corrected Stratum) ===\n\n")
cat("Studies per method:\n")
print(table(df_corrected$segmentation_method))

meta_seg <- metacont(
  data = df_corrected,
  n.e = n.e, mean.e = mean.e, sd.e = sd.e,
  n.c = n.c, mean.c = mean.c, sd.c = sd.c,
  studlab = studlab, sm = "SMD", method.tau = "REML",
  hakn = TRUE, comb.fixed = FALSE,
  subgroup = segmentation_method
)

cat("\n--- Overall (ICV-corrected) ---\n")
cat(sprintf("  k = %d, SMD = %.4f [%.4f, %.4f], I2 = %.1f%%\n",
            meta_seg$k, meta_seg$TE.random, meta_seg$lower.random,
            meta_seg$upper.random, meta_seg$I2 * 100))
cat(sprintf("  p (random) = %s\n\n", formatC(meta_seg$pval.random, format = "e", digits = 2)))

cat("--- Subgroup Results ---\n")
for (sg in unique(df_corrected$segmentation_method)) {
  idx <- which(meta_seg$subgroup == sg)
  k_sg <- length(idx)
  cat(sprintf("  %s (k=%d): SMD = %.4f [%.4f, %.4f], I2 = %.1f%%\n",
              sg, k_sg,
              meta_seg$TE.random.w[meta_seg$subgroup.levels == sg],
              meta_seg$lower.random.w[meta_seg$subgroup.levels == sg],
              meta_seg$upper.random.w[meta_seg$subgroup.levels == sg],
              meta_seg$I2.w[meta_seg$subgroup.levels == sg] * 100))
}

cat("\n--- Between-subgroup heterogeneity test ---\n")
cat(sprintf("  Q_between = %.4f, df = %d, p = %s\n",
            meta_seg$Q.b.random, meta_seg$df.Q.b.random,
            formatC(meta_seg$pval.Q.b.random, format = "e", digits = 4)))

if (meta_seg$pval.Q.b.random >= 0.05) {
  cat("\n  >> Segmentation method did NOT significantly moderate the effect (p >= 0.05)\n\n")
} else {
  cat("\n  >> Segmentation method SIGNIFICANTLY moderated the effect (p < 0.05)\n\n")
}

# --- Generate Master Summary Table ---
print("Step 5/5: Creating master summary table...")

master_table <- data.frame()

# Add Corrected group results
if(!is.null(results_corrected)) {
  corrected_summary <- results_corrected$summary_data %>%
    mutate(Method = "Corrected") %>%
    select(Method, Group, Category, SMD, Lower, Upper, P_val)
  master_table <- bind_rows(master_table, corrected_summary)
}

# Add Uncorrected group results
if(!is.null(results_uncorrected)) {
  uncorrected_summary <- results_uncorrected$summary_data %>%
    mutate(Method = "Uncorrected") %>%
    select(Method, Group, Category, SMD, Lower, Upper, P_val)
  master_table <- bind_rows(master_table, uncorrected_summary)
}

# Format and save
master_table <- master_table %>%
  mutate(
    CI_95 = paste0("[", round(Lower, 3), ", ", round(Upper, 3), "]"),
    SMD = round(SMD, 3),
    P_Value = sapply(P_val, fmt_p_exact),
    Significance = ifelse(P_val < 0.05, "Significant", "Non-significant")
  ) %>%
  select(Method, Group, Category, SMD, CI_95, P_Value, Significance) %>%
  arrange(Method, match(Category, c("Main", "Diagnosis", "Dose", "Bias")), Group)

write.csv(master_table, file.path(OUTPUT_FOLDER, "Table_Master_Summary_All_Results.csv"), row.names = FALSE)
print("Master summary table created!")
