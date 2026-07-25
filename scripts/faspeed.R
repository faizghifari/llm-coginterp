library(psych)

# swap in your actual R dataset object here
X <- read.csv("data/imputed/missforest/R/all_standard/imputed_model_benchmark_table.csv")

X <- X[, -1]  # drop identifier column
Rmat <- cor(X, use = "pairwise.complete.obs")

# time smc alone
t_smc <- system.time(smc_out <- smc(Rmat))
cat("smc():", t_smc["elapsed"], "s\n")

# time fa alone, profiled
Rprof("scripts/fa_profile.out")
t_fa <- system.time(
  fa_out <- fa(Rmat, nfactors = 20, fm = "minres", n.obs = nrow(X))
)
Rprof(NULL)
cat("fa():", t_fa["elapsed"], "s\n")

print(summaryRprof("scripts/fa_profile.out")$by.self[1:15, ])

cat("\nn iterations (optim converged?):", fa_out$fit, "\n")
cat("any Heywood cases (h2 >= .998):", sum(fa_out$communality >= .998), "\n")
