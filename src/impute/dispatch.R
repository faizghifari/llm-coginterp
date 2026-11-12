# ─────────────────────────────────────────────────────────────────────────────
# Method dispatcher shared by the imputation orchestrators (run/impute.R,
# run/impute_hier.R). Maps a method name to its impute_<method>() call, lazily
# sourcing the method file. Expects the caller's environment to define:
#   SRC      — the repo's src/ directory (method files live under impute/)
#   MAX_RANK — the shared hyperparameter cap (rank / ncp)
# ─────────────────────────────────────────────────────────────────────────────

impute_R <- function(method, x) {
  if (method == "softimpute") {
    source(file.path(SRC, "impute", "softimpute", "method.R"))
    impute_softimpute(x, max_rank = MAX_RANK)
  } else if (method == "softimpute_corr") {
    source(file.path(SRC, "impute", "corr_common.R"))
    source(file.path(SRC, "impute", "softimpute_corr", "method.R"))
    impute_softimpute_corr(x, max_rank = MAX_RANK)
  } else if (method == "optspace") {
    source(file.path(SRC, "impute", "corr_common.R"))
    source(file.path(SRC, "impute", "optspace", "method.R"))
    impute_optspace(x)
  } else if (method == "usvt") {
    source(file.path(SRC, "impute", "corr_common.R"))
    source(file.path(SRC, "impute", "usvt", "method.R"))
    impute_usvt(x)
  } else if (method == "iterativepca") {
    source(file.path(SRC, "impute", "iterativepca", "method.R"))
    impute_iterativepca(x, max_ncp = MAX_RANK)
  } else if (method == "knn") {
    source(file.path(SRC, "impute", "knn", "method.R"))
    impute_knn(x)
  } else if (method == "missforest") {
    source(file.path(SRC, "impute", "missforest", "method.R"))
    impute_missforest(x)
  } else if (method == "knn_corr") {
    source(file.path(SRC, "impute", "corr_common.R"))
    source(file.path(SRC, "impute", "knn_corr", "method.R"))
    impute_knn_corr(x)
  } else if (method == "missforest_corr") {
    source(file.path(SRC, "impute", "corr_common.R"))
    source(file.path(SRC, "impute", "missforest_corr", "method.R"))
    impute_missforest_corr(x)
  } else if (method == "mice") {
    source(file.path(SRC, "impute", "mice", "method.R"))
    impute_mice(x)
  } else if (method == "cvxr") {
    source(file.path(SRC, "impute", "corr_common.R"))
    source(file.path(SRC, "impute", "cvxr", "method.R"))
    impute_cvxr(x)
  } else if (method == "ggm") {
    source(file.path(SRC, "impute", "corr_common.R"))
    source(file.path(SRC, "impute", "ggm", "method.R"))
    impute_ggm(x)
  } else if (method == "default") {
    source(file.path(SRC, "impute", "corr_common.R"))
    source(file.path(SRC, "impute", "default", "method.R"))
    impute_default(x)
  } else if (method == "zeros") {
    source(file.path(SRC, "impute", "corr_common.R"))
    source(file.path(SRC, "impute", "zeros", "method.R"))
    impute_zeros(x)
  } else stop("not an R imputer: ", method)
}
