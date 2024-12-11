
MSMP <- read.csv("results_all_MSMP.csv")
MSMP_domain <- read.csv("results_all_domain_MSMP.csv")
domain_MSMP_blocking <- read.csv("results_all_domain_MSMP_blocking_test.csv")
MSMP_plus <- read.csv("results_all_MSMP_plus.csv")

MSMP_domain$fraction_of_comparisons_lsh <- MSMP_domain$lsh__N_c / ((1000*999)/2)
MSMP_domain$fraction_of_comparisons_clu <- MSMP_domain$clu__N_c / ((1000*999)/2)

MSMP$fraction_of_comparisons_lsh <- MSMP$lsh__N_c / ((1000*999)/2)
MSMP$fraction_of_comparisons_clu <- MSMP$clu__N_c / ((1000*999)/2)

domain_MSMP_blocking$fraction_of_comparisons_lsh <- domain_MSMP_blocking$lsh__N_c / ((1000*999)/2)
domain_MSMP_blocking$fraction_of_comparisons_clu <- domain_MSMP_blocking$clu__N_c / ((1000*999)/2)

MSMP_plus$fraction_of_comparisons_lsh <- MSMP_plus$lsh__N_c / ((1000*999)/2)
MSMP_plus$fraction_of_comparisons_clu <- MSMP_plus$clu__N_c / ((1000*999)/2)

library(ggplot2)
library(dplyr)

MSMP <- MSMP %>% mutate(source = "MSMP")
MSMP_domain <- MSMP_domain %>% mutate(source = "MSMP_domain")
domain_MSMP_blocking <- domain_MSMP_blocking %>% mutate(source = "domain_MSMP_blocking")
MSMP_plus <- MSMP_plus %>% mutate(source = "MSMP_plus")

combined_data <- bind_rows(MSMP, MSMP_domain, domain_MSMP_blocking, MSMP_plus)

combined_data <- combined_data %>% group_by(r, source) %>% summarise(fraction_of_comparisons_lsh = mean(fraction_of_comparisons_lsh), 
                                                                     fraction_of_comparisons_clu = mean(fraction_of_comparisons_clu),
                                                                     lsh__PQ = mean(lsh__PQ),
                                                                     lsh__PC = mean(lsh__PC),
                                                                     lsh__f1 = mean(lsh__f1), 
                                                                     clu__f1 = mean(clu__f1))

ggplot(combined_data, aes(x = fraction_of_comparisons_clu, y = clu__f1, color = source)) +
  geom_line() +
  labs(x = "Fraction of Comparisons",
    y = "F1-measure",
    color = "Source"
  ) +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 14),   # Increase size of axis tick labels
    axis.title = element_text(size = 16), # Increase size of axis titles
    legend.title = element_text(size = 14), # Increase size of legend title
    legend.text = element_text(size = 12)  # Increase size of legend text
  )


install.packages("pracma")
library(pracma)

auc_data <- combined_data %>%
  group_by(source) %>%
  arrange(fraction_of_comparisons_lsh, .by_group = TRUE) %>%
  summarise(
    auc = trapz(x = fraction_of_comparisons_lsh, y = clu__f1)
  )

print(auc_data)

