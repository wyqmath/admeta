# --- 最终完整代码 (v4.5 - 优化文件名 & 简化性别图) ---

# ===================================================================
# Part 0: 加载所需的程序包
# ===================================================================
library(tidyverse)
library(meta)
library(stringr)
library(data.table)
library(RColorBrewer)

# ===================================================================
# Part 1: 基础设置与数据预处理
# ===================================================================
# !!! 输入文件路径优先使用已设置路径；若找不到则在工作区中尝试常见替代路径。 ---
# 原始脚本使用了 Windows 专用路径；为便于在本工作区（macOS）运行，自动查找可用输入文件。
file_path <- "C:/Users/蔡旻诺/Desktop/apoe/data/icv_data.csv"
r <- 0.85
timestamp <- format(Sys.time(), "%Y-%m-%d_%H-%M")
folder_name_with_timestamp <- paste("Meta_Analysis_Results_Final", timestamp, sep = "_")

# 在几个常见位置查找输入文件，优先使用原始路径（若存在），否则使用工作区内的替代文件
possible_paths <- c(
  file_path,
  file.path(getwd(), "meta.csv"),
  file.path(getwd(), "Python_Meta_Results", "final_processed_data_python.csv"),
  file.path(dirname(getwd()), "ADNI", "ADNI_zong.csv"),
  file.path(dirname(getwd()), "NACC", "NACC_HV.csv")
)
file_path_found <- NULL
for(p in possible_paths) {
  if(file.exists(p)) { file_path_found <- p; break }
}
if(is.null(file_path_found)) {
  stop(paste("Input file not found. Tried:", paste(possible_paths, collapse = ", "))) 
} else {
  file_path <- file_path_found
}

# 将输出文件夹放在当前工作目录下，便于查看和保存结果
output_folder <- file.path(getwd(), folder_name_with_timestamp)

df_raw_with_ghost_cols <- fread(file_path)
df_raw <- df_raw_with_ghost_cols %>% select(where(~!all(is.na(.))))
core_numeric_cols <- c("apoe4_n", "apoe4_Mean", "apoe4_SD", "no_apoe4_n", "no_apoe4_Mean", "no_apoe4_SD", "age")
df_raw_cleaned <- df_raw %>% filter(if_all(all_of(core_numeric_cols), ~ !is.na(.)))
print(paste("原始数据共", nrow(df_raw), "行，在确保核心数值完整后，剩余", nrow(df_raw_cleaned), "行有效数据。"))


# ===================================================================
# Part 2: 数据整合
# ===================================================================
df_bilateral <- df_raw_cleaned %>% filter(hemisphere == "bilateral") %>% rename(studlab = title, n.e = apoe4_n, mean.e = apoe4_Mean, sd.e = apoe4_SD, n.c = no_apoe4_n, mean.c = no_apoe4_Mean, sd.c = no_apoe4_SD) %>% select(any_of(c("studlab", "diagnosis", "n.e", "mean.e", "sd.e", "n.c", "mean.c", "sd.c", "age", "correction_method", "year", "e4_dosage", "female_percentage")))
df_combined_lr <- df_raw_cleaned %>% filter(hemisphere %in% c("left", "right")) %>% pivot_wider(id_cols = any_of(c("title", "diagnosis", "apoe4_n", "no_apoe4_n", "age", "correction_method", "year", "e4_dosage", "female_percentage")), names_from = hemisphere, values_from = c(apoe4_Mean, apoe4_SD, no_apoe4_Mean, no_apoe4_SD)) %>% mutate(across(starts_with(c("apoe4_", "no_apoe4_")), as.numeric), mean_e_total = apoe4_Mean_left + apoe4_Mean_right, sd_e_total = sqrt(apoe4_SD_left^2 + apoe4_SD_right^2 + 2 * r * apoe4_SD_left * apoe4_SD_right), mean_c_total = no_apoe4_Mean_left + no_apoe4_Mean_right, sd_c_total = sqrt(no_apoe4_SD_left^2 + no_apoe4_SD_right^2 + 2 * r * no_apoe4_SD_left * no_apoe4_SD_right)) %>% rename(studlab = title, n.e = apoe4_n, mean.e = mean_e_total, sd.e = sd_e_total, n.c = no_apoe4_n, mean.c = mean_c_total, sd.c = sd_c_total) %>% select(any_of(c("studlab", "diagnosis", "n.e", "mean.e", "sd.e", "n.c", "mean.c", "sd.c", "age", "correction_method", "year", "e4_dosage", "female_percentage")))
df_final_dirty_labels <- bind_rows(df_bilateral, df_combined_lr)
df_final <- df_final_dirty_labels %>% mutate(studlab = str_replace_all(studlab, "[^A-Za-z0-9 ]", ""))
print("--- 数据整合与标签清理完成 ---")
print(paste("最终用于分析的总行数为:", nrow(df_final)))


# ===================================================================
# Part 3: 执行核心分析
# ===================================================================
meta_analysis <- metacont(data = df_final, n.e = n.e, mean.e = mean.e, sd.e = sd.e, n.c = n.c, mean.c = mean.c, sd.c = sd.c, studlab = studlab, sm = "SMD", method.tau = "REML", hakn = TRUE, comb.fixed = FALSE)
meta_regression_log_age <- metareg(meta_analysis, ~log(age))
bias_test <- metabias(meta_analysis, method.bias = "linreg", k.min = 10)
if("female_percentage" %in% names(df_final) && sum(!is.na(df_final$female_percentage)) > 1) { meta_regression_sex <- metareg(meta_analysis, ~female_percentage) } else { meta_regression_sex <- NULL }
print("--- 核心分析完成 ---")


# ===================================================================
# Part 4: 保存所有分析结果的图片
# ===================================================================
if (!dir.exists(output_folder)) { dir.create(output_folder) }
write.csv(df_final, file = file.path(output_folder, "final_processed_data.csv"), row.names = FALSE, fileEncoding = "UTF-8")
diagnosis <- df_final$diagnosis
png(filename = file.path(output_folder, "Forest_By_Diagnosis.png"), width = 10, height = 15, units = "in", res = 600); forest(meta_analysis, byvar = diagnosis, studlab = FALSE, comb.fixed = FALSE, print.byvar = TRUE, leftcols = c("n.e", "mean.e", "sd.e", "n.c", "mean.c", "sd.c"), leftlabs = c("N", "Mean", "SD", "N", "Mean", "SD"), rightcols = c("effect", "ci", "w.random"), rightlabs = c("SMD", "95%-CI", "Weight"), col.square = "royalblue", col.diamond = "maroon", cex = 0.7, label.e = "APOE ε4 (+)", label.c = "APOE ε4 (-)", just = "center"); dev.off()
df_ad <- df_final %>% filter(diagnosis == "AD"); if(nrow(df_ad)>1){ meta_ad <- metacont(data = df_ad, n.e = n.e, mean.e = mean.e, sd.e = sd.e, n.c = n.c, mean.c = mean.c, sd.c = sd.c, studlab = studlab, sm = "SMD", comb.fixed=FALSE); png(filename = file.path(output_folder, "Forest_Diagnosis_AD.png"), width = 10, height = 8, units = "in", res = 600); forest(meta_ad, studlab = FALSE, rightcols = c("effect", "ci", "w.random"), rightlabs = c("SMD", "95%-CI", "Weight"), leftcols = c("n.e", "mean.e", "sd.e", "n.c", "mean.c", "sd.c"), leftlabs = c("N", "Mean", "SD", "N", "Mean", "SD"), col.square = "darkred", col.diamond = "darkred", cex = 0.9, main = "Subgroup: AD", label.e = "APOE ε4 (+)", label.c = "APOE ε4 (-)", just = "center"); dev.off()}
df_mci <- df_final %>% filter(diagnosis == "MCI"); if(nrow(df_mci)>1){ meta_mci <- metacont(data = df_mci, n.e = n.e, mean.e = mean.e, sd.e = sd.e, n.c = n.c, mean.c = mean.c, sd.c = sd.c, studlab = studlab, sm = "SMD", comb.fixed=FALSE); png(filename = file.path(output_folder, "Forest_Diagnosis_MCI.png"), width = 10, height = 8, units = "in", res = 600); forest(meta_mci, studlab = FALSE, rightcols = c("effect", "ci", "w.random"), rightlabs = c("SMD", "95%-CI", "Weight"), leftcols = c("n.e", "mean.e", "sd.e", "n.c", "mean.c", "sd.c"), leftlabs = c("N", "Mean", "SD", "N", "Mean", "SD"), col.square = "darkgreen", col.diamond = "darkgreen", cex = 0.9, main = "Subgroup: MCI", label.e = "APOE ε4 (+)", label.c = "APOE ε4 (-)", just = "center"); dev.off()}
df_cn <- df_final %>% filter(diagnosis == "CN"); if(nrow(df_cn)>1){ meta_cn <- metacont(data = df_cn, n.e = n.e, mean.e = mean.e, sd.e = sd.e, n.c = n.c, mean.c = mean.c, sd.c = sd.c, studlab = studlab, sm = "SMD", comb.fixed=FALSE); png(filename = file.path(output_folder, "Forest_Diagnosis_CN.png"), width = 10, height = 8, units = "in", res = 600); forest(meta_cn, studlab = FALSE, rightcols = c("effect", "ci", "w.random"), rightlabs = c("SMD", "95%-CI", "Weight"), leftcols = c("n.e", "mean.e", "sd.e", "n.c", "mean.c", "sd.c"), leftlabs = c("N", "Mean", "SD", "N", "Mean", "SD"), col.square = "darkblue", col.diamond = "darkblue", cex = 0.9, main = "Subgroup: CN", label.e = "APOE ε4 (+)", label.c = "APOE ε4 (-)", just = "center"); dev.off()}
png(filename = file.path(output_folder, "Funnel_Overall.png"), width = 10, height = 8, units = "in", res = 600); funnel(meta_analysis, main = "Funnel Plot for Publication Bias"); dev.off()

# --- 生成带颜色的高清对数气泡图 (ggplot2) ---
plot_data_age_colored <- data.frame(TE = meta_analysis$TE, w.random = meta_analysis$w.random, age = df_final$age, diagnosis = df_final$diagnosis, studlab = df_final$studlab)
final_user_colors <- c("AD" = "#FBD4D1", "MCI" = "#F9F8D3", "CN" = "#D3EAF9")
bubble_plot_log_age_colored_gg <- ggplot(plot_data_age_colored, aes(x = age, y = TE)) + geom_point(aes(size = w.random, fill = diagnosis), shape = 21, alpha = 0.8) + geom_smooth(method = "lm", aes(weight = w.random), color = "black", se = FALSE) + scale_x_log10() + scale_fill_manual(values = final_user_colors) + labs(title = "Meta-Regression Bubble Plot by Age (Colored by Diagnosis)", x = "Average Age of Study Participants (Log Scale)", y = "Effect Size (SMD)", size = "Study Weight", fill = "Diagnosis") + theme_bw()
ggsave(filename = file.path(output_folder, "Bubble_MetaReg_Age.png"), plot = bubble_plot_log_age_colored_gg, width = 11, height = 8, dpi = 600)

if(!is.null(meta_regression_sex)){
  plot_data_sex <- data.frame(TE = meta_analysis$TE, w.random = meta_analysis$w.random, female_percentage = df_final$female_percentage, studlab = df_final$studlab) %>% filter(!is.na(female_percentage))
  bubble_plot_sex_gg <- ggplot(plot_data_sex, aes(x = female_percentage, y = TE)) + geom_point(aes(size = w.random), shape = 21, fill = "orchid", alpha = 0.7) + geom_smooth(method = "lm", aes(weight = w.random), color = "black", se = FALSE) + scale_x_log10(breaks = c(0.1, 0.2, 0.5, 1.0)) + labs(title = "Meta-Regression by Percentage of Female Participants", x = "Percentage of Female Participants (Log Scale)", y = "Effect Size (SMD)", size = "Study Weight") + theme_bw()
  ggsave(filename = file.path(output_folder, "Bubble_MetaReg_Sex_Log.png"), plot = bubble_plot_sex_gg, width = 10, height = 8, dpi = 600)
}

# ===================================================================
# Part 5: 创建并保存文献年份【堆积】柱状图
# ===================================================================
print("--- 正在创建文献年份堆积柱状图 ---")
year_diag_counts <- df_final %>% distinct(studlab, year, diagnosis) %>% count(year, diagnosis, name = "number_of_studies"); year_stacked_plot_final_user <- ggplot(data = year_diag_counts, aes(x = as.factor(year), y = number_of_studies, fill = diagnosis)) + geom_bar(stat = "identity", position = "stack", color = "black") + labs(title = "Distribution of Included Studies by Publication Year and Diagnosis", x = "Publication Year", y = "Number of Studies", fill = "Diagnosis") + scale_fill_manual(values = final_user_colors) + theme_classic(base_size = 14) + theme(plot.title = element_text(hjust = 0.5), axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "right" ); ggsave(filename = file.path(output_folder, "Bar_Publication_Year.png"), plot = year_stacked_plot_final_user, width = 12, height = 8, dpi = 600)
print("--- 文献年份堆积柱状图已保存。 ---")


# ===================================================================
# Part 6: 独立分析校正与未校正研究
# ===================================================================
print("--- 正在执行独立的校正/未校正亚组分析 ---")
df_corrected <- df_final %>% filter(correction_method != "None")
df_uncorrected <- df_final %>% filter(correction_method == "None")
print(paste("已分离数据：", nrow(df_corrected), "个已校正研究，", nrow(df_uncorrected), "个未校正研究。"))

meta_corrected <- NULL; meta_uncorrected <- NULL
if (nrow(df_corrected) > 1) {
  meta_corrected <- metacont(data = df_corrected, n.e = n.e, mean.e = mean.e, sd.e = sd.e, n.c = n.c, mean.c = mean.c, sd.c = sd.c, studlab = studlab, sm = "SMD", comb.fixed = FALSE, method.tau = "REML", hakn = TRUE)
  png(filename = file.path(output_folder, "Forest_Correction_Corrected.png"), width = 10, height = 8, units = "in", res = 600)
  forest(meta_corrected, studlab = FALSE, main = "Subgroup Analysis: Corrected Studies Only", leftcols = c("n.e", "mean.e", "sd.e", "n.c", "mean.c", "sd.c"), leftlabs = c("N", "Mean", "SD", "N", "Mean", "SD"), rightcols = c("effect", "ci", "w.random"), rightlabs = c("SMD", "95%-CI", "Weight"), col.square = "darkcyan", col.diamond = "darkcyan", cex = 0.9, label.e = "APOE ε4 (+)", label.c = "APOE ε4 (-)", just = "center")
  dev.off()
  print("--- 已校正研究的独立分析完成，森林图已保存。 ---")
} else { print("校正组研究数量不足（小于2），无法进行独立的荟萃分析。") }

if (nrow(df_uncorrected) > 1) {
  meta_uncorrected <- metacont(data = df_uncorrected, n.e = n.e, mean.e = mean.e, sd.e = sd.e, n.c = n.c, mean.c = mean.c, sd.c = sd.c, studlab = studlab, sm = "SMD", comb.fixed = FALSE, method.tau = "REML", hakn = TRUE)
  png(filename = file.path(output_folder, "Forest_Correction_Uncorrected.png"), width = 10, height = 8, units = "in", res = 600)
  forest(meta_uncorrected, studlab = FALSE, main = "Subgroup Analysis: Uncorrected Studies Only", leftcols = c("n.e", "mean.e", "sd.e", "n.c", "mean.c", "sd.c"), leftlabs = c("N", "Mean", "SD", "N", "Mean", "SD"), rightcols = c("effect", "ci", "w.random"), rightlabs = c("SMD", "95%-CI", "Weight"), col.square = "chocolate", col.diamond = "chocolate", cex = 0.9, label.e = "APOE ε4 (+)", label.c = "APOE ε4 (-)", just = "center")
  dev.off()
  print("--- 未校正研究的独立分析完成，森林图已保存。 ---")
} else { print("未校正组研究数量不足（小于2），无法进行独立的荟萃分析。") }

if (!is.null(meta_corrected) && !is.null(meta_uncorrected)) {
  format_p_value <- function(p_val) { if (is.na(p_val)) { return(NA) } else if (p_val < 0.001) { return("<0.001") } else { return(as.character(round(p_val, 4))) } }
  k_corr <- meta_corrected$k; smd_corr <- round(meta_corrected$TE.random, 4); ci_lower_corr <- round(meta_corrected$lower.random, 4); ci_upper_corr <- round(meta_corrected$upper.random, 4); p_corr <- format_p_value(meta_corrected$pval.random); i2_corr <- paste0(round(meta_corrected$I2 * 100, 1), "%"); tau2_corr <- round(meta_corrected$tau2, 4)
  k_uncorr <- meta_uncorrected$k; smd_uncorr <- round(meta_uncorrected$TE.random, 4); ci_lower_uncorr <- round(meta_uncorrected$lower.random, 4); ci_upper_uncorr <- round(meta_uncorrected$upper.random, 4); p_uncorr <- format_p_value(meta_uncorrected$pval.random); i2_uncorr <- paste0(round(meta_uncorrected$I2 * 100, 1), "%"); tau2_uncorr <- round(meta_uncorrected$tau2, 4)
  correction_comparison_table <- data.frame(Indicator = c("Number of Studies (k)", "Pooled SMD", "95% CI", "p-value", "Heterogeneity (I^2)", "Between-Study Variance (tau^2)"), `Corrected Studies` = c(k_corr, smd_corr, paste0("[", ci_lower_corr, "; ", ci_upper_corr, "]"), p_corr, i2_corr, tau2_corr), `Uncorrected Studies` = c(k_uncorr, smd_uncorr, paste0("[", ci_lower_uncorr, "; ", ci_upper_uncorr, "]"), p_uncorr, i2_uncorr, tau2_uncorr))
  print("--- 校正方法独立分析结果对比 ---"); print(correction_comparison_table)
  write.csv(correction_comparison_table, file = file.path(output_folder, "comparison_by_correction_method.csv"), row.names = FALSE, fileEncoding = "UTF-8")
} else { print("由于一个或两个亚组数据不足，无法生成对比表格。") }


# ===================================================================
# Part 6.1: 细分校正方法 (Vicv vs. Sicv) 的独立分析
# ===================================================================
print("--- 正在按具体的校正方法 (Vicv vs. Sicv) 进行细分分析 ---")
df_vicv <- df_final %>% filter(correction_method == "Vicv")
df_sicv <- df_final %>% filter(correction_method == "Sicv")
print(paste("已分离数据：", nrow(df_vicv), "个 Vicv 校正研究，", nrow(df_sicv), "个 Sicv 校正研究。"))

meta_vicv <- NULL; meta_sicv <- NULL
if (nrow(df_vicv) > 1) {
  meta_vicv <- metacont(data = df_vicv, n.e = n.e, mean.e = mean.e, sd.e = sd.e, n.c = n.c, mean.c = mean.c, sd.c = sd.c, studlab = studlab, sm = "SMD", comb.fixed = FALSE, method.tau = "REML", hakn = TRUE)
  png(filename = file.path(output_folder, "Forest_CorrectionMethod_Vicv.png"), width = 10, height = 8, units = "in", res = 600)
  forest(meta_vicv, studlab = FALSE, main = "Subgroup: Vicv Corrected Studies", leftcols = c("n.e", "mean.e", "sd.e", "n.c", "mean.c", "sd.c"), leftlabs = c("N", "Mean", "SD", "N", "Mean", "SD"), rightcols = c("effect", "ci", "w.random"), rightlabs = c("SMD", "95%-CI", "Weight"), col.square = "darkmagenta", col.diamond = "darkmagenta", cex = 0.9, label.e = "APOE ε4 (+)", label.c = "APOE ε4 (-)", just = "center")
  dev.off()
  print("--- Vicv 校正组的独立分析完成，森林图已保存。 ---")
} else { print("Vicv 校正组研究数量不足（小于2），无法进行独立的荟萃分析。") }

if (nrow(df_sicv) > 1) {
  meta_sicv <- metacont(data = df_sicv, n.e = n.e, mean.e = mean.e, sd.e = sd.e, n.c = n.c, mean.c = mean.c, sd.c = sd.c, studlab = studlab, sm = "SMD", comb.fixed = FALSE, method.tau = "REML", hakn = TRUE)
  png(filename = file.path(output_folder, "Forest_CorrectionMethod_Sicv.png"), width = 10, height = 8, units = "in", res = 600)
  forest(meta_sicv, studlab = FALSE, main = "Subgroup: Sicv Corrected Studies", leftcols = c("n.e", "mean.e", "sd.e", "n.c", "mean.c", "sd.c"), leftlabs = c("N", "Mean", "SD", "N", "Mean", "SD"), rightcols = c("effect", "ci", "w.random"), rightlabs = c("SMD", "95%-CI", "Weight"), col.square = "darkslateblue", col.diamond = "darkslateblue", cex = 0.9, label.e = "APOE ε4 (+)", label.c = "APOE ε4 (-)", just = "center")
  dev.off()
  print("--- Sicv 校正组的独立分析完成，森林图已保存。 ---")
} else { print("Sicv 校正组研究数量不足（小于2），无法进行独立的荟萃分析。") }

if (!is.null(meta_vicv) && !is.null(meta_sicv)) {
  k_vicv <- meta_vicv$k; smd_vicv <- round(meta_vicv$TE.random, 4); ci_lower_vicv <- round(meta_vicv$lower.random, 4); ci_upper_vicv <- round(meta_vicv$upper.random, 4); p_vicv <- format_p_value(meta_vicv$pval.random); i2_vicv <- paste0(round(meta_vicv$I2 * 100, 1), "%"); tau2_vicv <- round(meta_vicv$tau2, 4)
  k_sicv <- meta_sicv$k; smd_sicv <- round(meta_sicv$TE.random, 4); ci_lower_sicv <- round(meta_sicv$lower.random, 4); ci_upper_sicv <- round(meta_sicv$upper.random, 4); p_sicv <- format_p_value(meta_sicv$pval.random); i2_sicv <- paste0(round(meta_sicv$I2 * 100, 1), "%"); tau2_sicv <- round(meta_sicv$tau2, 4)
  method_comparison_table <- data.frame(Indicator = c("Number of Studies (k)", "Pooled SMD", "95% CI", "p-value", "Heterogeneity (I^2)", "Between-Study Variance (tau^2)"), `Vicv Corrected` = c(k_vicv, smd_vicv, paste0("[", ci_lower_vicv, "; ", ci_upper_vicv, "]"), p_vicv, i2_vicv, tau2_vicv), `Sicv Corrected` = c(k_sicv, smd_sicv, paste0("[", ci_lower_sicv, "; ", ci_upper_sicv, "]"), p_sicv, i2_sicv, tau2_sicv))
  print("--- 具体校正方法 (Vicv vs Sicv) 结果对比 ---"); print(method_comparison_table)
  write.csv(method_comparison_table, file = file.path(output_folder, "comparison_by_VICV_SICV_method.csv"), row.names = FALSE, fileEncoding = "UTF-8")
} else { print("由于一个或两个具体校正方法亚组数据不足，无法生成对比表格。") }


# ===================================================================
# Part 6.2: 打印细分校正方法模型的完整摘要
# ===================================================================
if (!is.null(meta_vicv)) { print("--- Vicv 模型完整摘要 ---"); summary(meta_vicv) }
if (!is.null(meta_sicv)) { print("--- Sicv 模型完整摘要 ---"); summary(meta_sicv) }


# ===================================================================
# Part 7: 敏感性分析
# ===================================================================
print("--- 正在进行敏感性分析 ---")
df_with_effects <- as.data.frame(meta_analysis); outlier_studlabs <- df_with_effects %>% filter(TE > 1.0) %>% pull(studlab); df_sensitivity <- df_final %>% filter(!studlab %in% outlier_studlabs)
print(paste("已自动识别并排除了", length(outlier_studlabs), "个异常值研究: ", paste(outlier_studlabs, collapse=", ")))
meta_sensitivity <- metacont(data = df_sensitivity, n.e = n.e, mean.e = mean.e, sd.e = sd.e, n.c = n.c, mean.c = mean.c, sd.c = sd.c, studlab = studlab, sm = "SMD", comb.fixed = FALSE, method.tau = "REML", hakn = TRUE)
png(filename = file.path(output_folder, "Forest_Sensitivity.png"), width = 10, height = 12, units = "in", res = 600); forest(meta_sensitivity, studlab = FALSE, rightcols = c("effect", "ci", "w.random"), rightlabs = c("SMD", "95%-CI", "Weight"), leftcols = c("n.e", "mean.e", "sd.e", "n.c", "mean.c", "sd.c"), leftlabs = c("N", "Mean", "SD", "N", "Mean", "SD"), cex = 0.7, col.square = "seagreen", col.diamond = "darkgreen", label.e = "APOE ε4 (+)", label.c = "APOE ε4 (-)", just = "center"); dev.off()
print("--- 敏感性分析模型已计算完成。 ---")


# ===================================================================
# Part 8: 自动生成敏感性分析对比表格
# ===================================================================
print("--- 正在生成敏感性分析对比表格 ---")
k_main <- meta_analysis$k; smd_main <- round(meta_analysis$TE.random, 4); ci_lower_main <- round(meta_analysis$lower.random, 4); ci_upper_main <- round(meta_analysis$upper.random, 4); p_main <- format_p_value(meta_analysis$pval.random); i2_main <- paste0(round(meta_analysis$I2 * 100, 1), "%"); tau2_main <- round(meta_analysis$tau2, 4)
k_sens <- meta_sensitivity$k; smd_sens <- round(meta_sensitivity$TE.random, 4); ci_lower_sens <- round(meta_sensitivity$lower.random, 4); ci_upper_sens <- round(meta_sensitivity$upper.random, 4); p_sens <- format_p_value(meta_sensitivity$pval.random); i2_sens <- paste0(round(meta_sensitivity$I2 * 100, 1), "%"); tau2_sens <- round(meta_sensitivity$tau2, 4)
comparison_table <- data.frame(
  Indicator = c("Number of Studies (k)", "Pooled SMD", "95% CI", "p-value", "Heterogeneity (I^2)", "Between-Study Variance (tau^2)"),
  `Primary Analysis (all studies)` = c(k_main, smd_main, paste0("[", ci_lower_main, "; ", ci_upper_main, "]"), p_main, i2_main, tau2_main),
  `Sensitivity Analysis (outlier excluded)` = c(k_sens, smd_sens, paste0("[", ci_lower_sens, "; ", ci_upper_sens, "]"), p_sens, i2_sens, tau2_sens)
)
print(comparison_table)


# ===================================================================
# Part 9: APOE4 剂量效应分析 (总体分析)
# ===================================================================
print("--- 正在进行 APOE4 剂量效应分析 (总体) ---")
df_dosage <- df_final %>% filter(!is.na(e4_dosage), e4_dosage %in% c(1, 2))

df_homozygotes <- df_dosage %>% filter(e4_dosage == 2)
if(nrow(df_homozygotes) > 1) { 
  meta_homozygotes <- metacont(data = df_homozygotes, n.e = n.e, mean.e = mean.e, sd.e = sd.e, n.c = n.c, mean.c = mean.c, sd.c = sd.c, studlab = studlab, sm = "SMD", comb.fixed = FALSE) 
} else { meta_homozygotes <- NULL }

df_heterozygotes <- df_dosage %>% filter(e4_dosage == 1)
if(nrow(df_heterozygotes) > 1) { 
  meta_heterozygotes <- metacont(data = df_heterozygotes, n.e = n.e, mean.e = mean.e, sd.e = sd.e, n.c = n.c, mean.c = mean.c, sd.c = sd.c, studlab = studlab, sm = "SMD", comb.fixed = FALSE) 
} else { meta_heterozygotes <- NULL }

if(!is.null(meta_heterozygotes) && !is.null(meta_homozygotes)) {
  k_homo <- meta_homozygotes$k; smd_homo <- round(meta_homozygotes$TE.random, 2); ci_lower_homo <- round(meta_homozygotes$lower.random, 2); ci_upper_homo <- round(meta_homozygotes$upper.random, 2); i2_homo <- paste0(round(meta_homozygotes$I2 * 100, 1), "%"); p_homo <- format_p_value(meta_homozygotes$pval.random)
  k_het <- meta_heterozygotes$k; smd_het <- round(meta_heterozygotes$TE.random, 2); ci_lower_het <- round(meta_heterozygotes$lower.random, 2); ci_upper_het <- round(meta_heterozygotes$upper.random, 2); i2_het <- paste0(round(meta_heterozygotes$I2 * 100, 1), "%"); p_het <- format_p_value(meta_heterozygotes$pval.random)
  
  dosage_comparison_table <- data.frame(
    Indicator = c("Number of Studies (k)", "Pooled SMD", "95% CI", "p-value", "Heterogeneity (I^2)"),
    `APOE ε4 Homozygotes` = c(k_homo, smd_homo, paste0("[", ci_lower_homo, "; ", ci_upper_homo, "]"), p_homo, i2_homo),
    `APOE ε4 Heterozygotes` = c(k_het, smd_het, paste0("[", ci_lower_het, "; ", ci_upper_het, "]"), p_het, i2_het)
  )
  print("--- (总体) APOE4 剂量效应对对比表格 ---"); print(dosage_comparison_table)
  
  png(filename = file.path(output_folder, "Forest_Dosage_Homozygotes_Overall.png"), width = 10, height = 8, units = "in", res = 600); forest(meta_homozygotes, studlab = FALSE, main = "APOE ε4 Homozygotes vs. Non-carriers", rightcols = c("effect", "ci", "w.random"), rightlabs = c("SMD", "95%-CI", "Weight"), leftcols = c("n.e", "mean.e", "sd.e", "n.c", "mean.c", "sd.c"), leftlabs = c("N", "Mean", "SD", "N", "Mean", "SD"), col.square = "firebrick", col.diamond = "firebrick", cex = 0.9, label.e = "APOE ε4 Homozygotes", label.c = "Non-carriers", just = "center"); dev.off()
  png(filename = file.path(output_folder, "Forest_Dosage_Heterozygotes_Overall.png"), width = 10, height = 8, units = "in", res = 600); forest(meta_heterozygotes, studlab = FALSE, main = "APOE ε4 Heterozygotes vs. Non-carriers", rightcols = c("effect", "ci", "w.random"), rightlabs = c("SMD", "95%-CI", "Weight"), leftcols = c("n.e", "mean.e", "sd.e", "n.c", "mean.c", "sd.c"), leftlabs = c("N", "Mean", "SD", "N", "Mean", "SD"), col.square = "darkorange", col.diamond = "darkorange", cex = 0.9, label.e = "APOE ε4 Heterozygotes", label.c = "Non-carriers", just = "center"); dev.off()
} else {
  print("数据不足，无法同时进行纯合子与杂合子的总体剂量效应分析。")
}
print("--- 总体剂量效应分析已完成。 ---")


# ===================================================================
# Part 9.1 (升级版): 按诊断亚组进行剂量效应分析并绘制森林图
# ===================================================================
print("--- 正在按诊断亚组 (AD, MCI, CN) 进行剂量效应分析 ---")
subgroup_dosage_tables <- list()
for (dx in c("AD", "MCI", "CN")) {
  print(paste("--- 分析亚组:", dx, "---"))
  df_dx_dosage <- df_dosage %>% filter(diagnosis == dx)
  if(nrow(df_dx_dosage) == 0) { print("该亚组无剂量数据。"); next }
  
  df_dx_homo <- df_dx_dosage %>% filter(e4_dosage == 2)
  df_dx_het <- df_dx_dosage %>% filter(e4_dosage == 1)
  meta_dx_homo <- NULL; meta_dx_het <- NULL
  
  if(nrow(df_dx_homo) > 1) {
    meta_dx_homo <- metacont(data = df_dx_homo, studlab = studlab, n.e = n.e, mean.e = mean.e, sd.e = sd.e, n.c = n.c, mean.c = mean.c, sd.c = sd.c, sm = "SMD", comb.fixed = FALSE)
    png(filename = file.path(output_folder, paste0("Forest_Dosage_Homozygotes_", dx, ".png")), width = 10, height = 8, units = "in", res = 600)
    forest(meta_dx_homo, studlab = FALSE, main = paste("APOE ε4 Homozygotes vs. Non-carriers (Subgroup:", dx, ")"), rightcols = c("effect", "ci", "w.random"), rightlabs = c("SMD", "95%-CI", "Weight"), leftcols = c("n.e", "mean.e", "sd.e", "n.c", "mean.c", "sd.c"), leftlabs = c("N", "Mean", "SD", "N", "Mean", "SD"), col.square = "firebrick", col.diamond = "firebrick", cex = 0.9, label.e = "APOE ε4 Homozygotes", label.c = "Non-carriers", just = "center")
    dev.off()
    print(paste("---", dx, "亚组: 纯合子森林图已保存。 ---"))
  } else { print(paste(dx, "亚组: 纯合子研究数量不足 (", nrow(df_dx_homo), ")。")) }
  
  if(nrow(df_dx_het) > 1) {
    meta_dx_het <- metacont(data = df_dx_het, studlab = studlab, n.e = n.e, mean.e = mean.e, sd.e = sd.e, n.c = n.c, mean.c = mean.c, sd.c = sd.c, sm = "SMD", comb.fixed = FALSE)
    png(filename = file.path(output_folder, paste0("Forest_Dosage_Heterozygotes_", dx, ".png")), width = 10, height = 8, units = "in", res = 600)
    forest(meta_dx_het, studlab = FALSE, main = paste("APOE ε4 Heterozygotes vs. Non-carriers (Subgroup:", dx, ")"), rightcols = c("effect", "ci", "w.random"), rightlabs = c("SMD", "95%-CI", "Weight"), leftcols = c("n.e", "mean.e", "sd.e", "n.c", "mean.c", "sd.c"), leftlabs = c("N", "Mean", "SD", "N", "Mean", "SD"), col.square = "darkorange", col.diamond = "darkorange", cex = 0.9, label.e = "APOE ε4 Heterozygotes", label.c = "Non-carriers", just = "center")
    dev.off()
    print(paste("---", dx, "亚组: 杂合子森林图已保存。 ---"))
  } else { print(paste(dx, "亚组: 杂合子研究数量不足 (", nrow(df_dx_het), ")。")) }
  
  if(!is.null(meta_dx_homo) && !is.null(meta_dx_het)) {
    k_homo <- meta_dx_homo$k; smd_homo <- round(meta_dx_homo$TE.random, 2); ci_lower_homo <- round(meta_dx_homo$lower.random, 2); ci_upper_homo <- round(meta_dx_homo$upper.random, 2); i2_homo <- paste0(round(meta_dx_homo$I2 * 100, 1), "%"); p_homo <- format_p_value(meta_dx_homo$pval.random)
    k_het <- meta_dx_het$k; smd_het <- round(meta_dx_het$TE.random, 2); ci_lower_het <- round(meta_dx_het$lower.random, 2); ci_upper_het <- round(meta_dx_het$upper.random, 2); i2_het <- paste0(round(meta_dx_het$I2 * 100, 1), "%"); p_het <- format_p_value(meta_dx_het$pval.random)
    
    sub_dosage_table <- data.frame(
      Indicator = c("Number of Studies (k)", "Pooled SMD", "95% CI", "p-value", "Heterogeneity (I^2)"),
      `APOE ε4 Homozygotes` = c(k_homo, smd_homo, paste0("[", ci_lower_homo, "; ", ci_upper_homo, "]"), p_homo, i2_homo),
      `APOE ε4 Heterozygotes` = c(k_het, smd_het, paste0("[", ci_lower_het, "; ", ci_upper_het, "]"), p_het, i2_het)
    )
    
    print(paste("--- 剂量效应分析对比表格 (亚组:", dx, ") ---"))
    print(sub_dosage_table)
    subgroup_dosage_tables[[dx]] <- sub_dosage_table
  } else {
    print(paste("在", dx, "亚组中，数据不足以同时比较纯合子与杂合子。"))
  }
}
print("--- 所有诊断亚组的剂量效应分析已完成。 ---")


# ===================================================================
# Part 10: 保存完整的统计报告 (已更新)
# ===================================================================
final_summary_text <- capture.output({
  print(" "); print(paste("--- 荟萃分析完整报告 (", timestamp, ") ---", sep="")); print(" ");
  print("--- 1. 主荟萃分析结果 (Overall Analysis) ---"); print(summary(meta_analysis, common = FALSE));
  print(" "); print("--- 2. 亚组分析详情 (By Diagnosis) ---");
  print("--- 2a. AD 亚组 ---"); if(!is.null(meta_ad)) { print(summary(meta_ad, common = FALSE)) } else { print("数据不足。") }
  print("--- 2b. MCI 亚组 ---"); if(!is.null(meta_mci)) { print(summary(meta_mci, common = FALSE)) } else { print("数据不足。") }
  print("--- 2c. CN 亚组 ---"); if(!is.null(meta_cn)) { print(summary(meta_cn, common = FALSE)) } else { print("数据不足。") }
  print(" "); print("--- 3. 元回归分析 ---");
  print("--- 3a. 年龄元回归 ---"); print(summary(meta_regression_log_age));
  if(!is.null(meta_regression_sex)){ print(" "); print("--- 3b. 性别比例元回归 ---"); print(summary(meta_regression_sex)); }
  print(" "); print("--- 4. 发表偏倚检验 ---"); print(bias_test);
  print(" "); print("--- 5. 敏感性分析 ---"); print(comparison_table);
  print(" "); print("--- 6. 校正方法亚组分析 ---"); if(exists("correction_comparison_table")){print(correction_comparison_table)} else {print("数据不足")}
  print(" "); print("--- 7. 具体校正方法 (Vicv vs Sicv) 亚组分析 ---"); if(exists("method_comparison_table")){print(method_comparison_table)} else {print("数据不足")}
  print(" "); print("--- 8. APOE4 剂量效应分析 (总体) ---"); if(exists("dosage_comparison_table")){print(dosage_comparison_table)} else {print("数据不足")}
  
  print(" "); print("--- 9. APOE4 剂量效应分析 (按诊断亚组) ---");
  if(length(subgroup_dosage_tables) > 0) {
    for(dx in names(subgroup_dosage_tables)) {
      print(paste("--- 亚组:", dx, "---"))
      print(subgroup_dosage_tables[[dx]])
    }
  } else {
    print("所有亚组均数据不足，无法生成剂量效应表格。")
  }
})
writeLines(final_summary_text, con = file.path(output_folder, "statistical_summary_complete.txt"))
print("--- 完整的、包含所有亚组详情的统计报告已保存。 ---")


# ===================================================================
# Part 11: 最终成功信息
# ===================================================================
print(paste("所有结果已成功保存至:", output_folder))