# Pacotes ------------------------------------------------------------------
library(tidymodels)
library(tidyverse)
library(pROC)
library(vip)
library(themis)

# Bases de dados ----------------------------------------------------------
# Base de treino e teste --------------------------------------------------
porto_train <- read_csv("porto_train.csv")|>
  mutate(target = as.character(target))

porto_test <- read_csv("porto_test.csv")

glimpse(porto_train)

porto_train |> count(target) # desfecho não esta balanceado!

# Reamostragem ------------------------------------------------------------

porto_resamples <- vfold_cv(porto_train, v = 2, strata = "target")

# Exploratória ------------------------------------------------------------

skimr::skim(porto_train)

# XGBoost -----------------------------------------------------------------

## Data prep

porto_recipe <-
  recipe(target ~ ., data = porto_train)|>
  step_downsample(under_ratio = 1,
                  column = "id") |>
  step_normalize(all_numeric_predictors()) |>
  # step_novel(all_nominal_predictors()) |>
  # step_impute_linear(Income, Assets, Debt, impute_with = imp_vars(Expenses)) %>%
  # step_impute_mode(all_nominal_predictors()) |>
  # step_zv(all_predictors()) |>
  # step_poly(all_numeric_predictors(), degree = 9) |>
  step_corr(all_numeric_predictors(),threshold = 0.7)



## Estratégia de Tunagem de Hiperparâmetros

# Achar uma combinação `learning_rate` e `trees` que funciona relativamente bem. Usando uma learning_rate alta. Vamos fixar os valores dos outros parâmetros.
#
# -   `min_n`: usar um valor entre 1 e 30 é razoável no começo.
# -   `max_depth`: geralmente começamos com algo entre 4 e 6.
# -   `loss_reduction`: vamos começar com 0, geralmente começamos com valores baixos.
# -   `mtry`: começamos com +- 80% do número de colunas na base.
# -   `sample_size`: também fazemos approx 80% do número de linhas.
#
# Em seguida vamos tunar o `learn_rate` e `trees` em um grid assim:
#
# -   `learn_rate` - 0.05, 0.1, 0.3

# -   `trees` - 100, 500, 1000, 1500

# ### Passo 1:

# Achar uma combinação `learning_rate` e `trees` que funciona relativamente bem. Usando uma learning_rate alta.
# Vamos fixar os valores dos outros parâmetros.

## Modelo

porto_model <-
  boost_tree(
  min_n = 15,
  mtry = 0.8,
  trees = tune(),
  tree_depth = 4,
  learn_rate = tune(),
  loss_reduction = 0,
  sample_size = 0.8
) |>
  set_mode("classification") |>
  set_engine("xgboost", count = FALSE)

## Workflow

porto_wf <- workflow() |>
  add_model(porto_model) |>
  add_recipe(porto_recipe)

## Tune grid para learn rate e trees

porto_grid <- expand.grid(
  trees = c(50, 150, 300, 500),
  learn_rate = c(0.03, 0.3)
)

porto_grid

# doParallel::registerDoParallel(4)

porto_tune_grid <-
  tune_grid(
    porto_wf,
    resamples = porto_resamples,
    grid = porto_grid,
    metrics = metric_set(roc_auc)
)

#### Melhores hiperparâmetros

autoplot(porto_tune_grid)

show_best(porto_tune_grid, n = 6)

porto_select_best_passo1 <-
  porto_tune_grid |>
  select_best(metric = "roc_auc")

porto_select_best_passo1

## Passo 2

porto_select_best_passo1$trees
porto_select_best_passo1$learn_rate

# Vimos que com os parâmetros da árvore fixos:
#
# -   `trees` = `r telco_select_best_passo1$trees`
# -   `learn_rate` = `r telco_select_best_passo1$learn_rate`
#
# São bons valores inciais. Agora, podemos tunar os parâmetros relacionados à árvore.
#
# -   `tree_depth`: vamos deixar ele variar entre 3 e 10.
# -   `min_n`: vamos deixar variar entre 5 e 90.
#
# Os demais deixamos fixos como anteriormente.

## Modelo

base_xb_model <- boost_tree(
  min_n = tune(),
  mtry = 0.8,
  trees = base_select_xgb_best_passo1$trees,
  tree_depth = tune(),
  learn_rate = base_select_xgb_best_passo1$learn_rate,
  loss_reduction = 0,
  sample_size = 0.8
) |>
  set_mode("classification") |>
  set_engine("xgboost", count = FALSE)

## Workflow

base_xb_wf <- workflow() |>
  add_model(base_xb_model) |>
  add_recipe(base_xb_recipe)

## Tune grid para learn rate e trees

grid_xb <- expand.grid(
  tree_depth = c(3, 4, 6),
  min_n = c(30, 60, 90)
)

grid_xb

# doParallel::registerDoParallel(4)

base_xb_tune_grid <- tune_grid(
  base_xb_wf,
  resamples = base_resamples,
  grid = grid_xb,
  metrics = metric_set(roc_auc)
)

#### Melhores hiperparâmetros

autoplot(base_xb_tune_grid)
show_best(base_xb_tune_grid, n = 6)

base_select_xgb_best_passo2 <- base_xb_tune_grid |>
  select_best(metric = "roc_auc")

base_select_xgb_best_passo2

### Passo 3:

# Agora também temos definidos:

base_select_xgb_best_passo2$tree_depth
base_select_xgb_best_passo2$min_n

# -   `trees` = `r telco_select_best_passo1$trees`
# -   `learn_rate` = `r telco_select_best_passo1$learn_rate`
# -   `min_n` = `r telco_select_best_passo2$min_n`
# -   `tree_depth` = `r telco_select_best_passo2$tree_depth`
#
# Vamos então tunar o `loss_reduction`:
#
#   `loss_reduction`: vamos deixar ele variar entre 0 e 2

## Modelo

base_xb_model <- boost_tree(
  min_n = base_select_xgb_best_passo2$min_n,
  mtry = 0.8,
  trees = base_select_xgb_best_passo1$trees,
  tree_depth = base_select_xgb_best_passo2$tree_depth,
  learn_rate = base_select_xgb_best_passo1$learn_rate,
  loss_reduction = tune(),
  sample_size = 0.8
) |>
  set_mode("classification") |>
  set_engine("xgboost", count = FALSE)

## Workflow

base_xb_wf <- workflow() |>
  add_model(base_xb_model) |>
  add_recipe(base_xb_recipe)

## Tune grid para learn rate e trees

grid_xb <- expand.grid(
  loss_reduction = c(0, 0.05, 1, 1.5, 2)
)

grid_xb

# doParallel::registerDoParallel(4)

base_xb_tune_grid <- tune_grid(
  base_xb_wf,
  resamples = base_resamples,
  grid = grid_xb,
  metrics = metric_set(roc_auc)
)

#### Melhores hiperparâmetros

autoplot(base_xb_tune_grid)
show_best(base_xb_tune_grid, n = 6)

base_select_xgb_best_passo3 <- base_xb_tune_grid |>
  select_best(metric = "roc_auc")

base_select_xgb_best_passo3

### Passo 4:

# Não parece que o `lossreduction` teve tanto efeito, mas, vamos usar "base_select_xgb_best_passo3$loss_reduction" que deu o melhor resultado. Até agora temos definido:

base_select_xgb_best_passo3$loss_reduction

# -   `trees` = `r telco_select_best_passo1$trees`
# -   `learn_rate` = `r telco_select_best_passo1$learn_rate`
# -   `min_n` = `r telco_select_best_passo2$min_n`
# -   `tree_depth` = `r telco_select_best_passo2$tree_depth`
# -   `lossreduction` = `r telco_select_best_passo3$loss_reduction`
#
# Vamos então tunar o `mtry` e o `sample_size`:
#
#   -   `mtry`: de 10% a 100%
#   -   `sample_size`: de 50% a 100%

## Modelo

base_xb_model <- boost_tree(
  min_n = base_select_xgb_best_passo2$min_n,
  mtry = tune(),
  trees = base_select_xgb_best_passo1$trees,
  tree_depth = base_select_xgb_best_passo2$tree_depth,
  learn_rate = base_select_xgb_best_passo1$learn_rate,
  loss_reduction = base_select_xgb_best_passo3$loss_reduction,
  sample_size = tune()
) |>
  set_mode("classification") |>
  set_engine("xgboost", count = FALSE)

## Workflow

base_xb_wf <- workflow() |>
  add_model(base_xb_model) |>
  add_recipe(base_xb_recipe)

## Tune grid para learn rate e trees

grid_xb <- expand.grid(
  sample_size = seq(0.5, 1.0, length.out = 2),
  mtry = seq(0.1, 1.0, length.out = 2)
)

grid_xb

# doParallel::registerDoParallel(4)

base_xb_tune_grid <- tune_grid(
  base_xb_wf,
  resamples = base_resamples,
  grid = grid_xb,
  metrics = metric_set(roc_auc)
)

#### Melhores hiperparâmetros

autoplot(base_xb_tune_grid)
show_best(base_xb_tune_grid, n = 6)

base_select_xgb_best_passo4 <- base_xb_tune_grid |>
  select_best(metric = "roc_auc")

base_select_xgb_best_passo4


### Passo 5:

base_select_xgb_best_passo4$mtry

base_select_xgb_best_passo4$sample_size

# Vimos que a melhor combinação foi
#
# -   `mtry` = `r telco_select_best_passo4$mtry`
# -   `sample_size` = `r telco_select_best_passo4$sample_size`
#
# Agora vamos tunar o `learn_rate` e o `trees` de novo, mas deixando o `learn_rate` assumir valores menores.

## Modelo

base_xb_model <- boost_tree(
  min_n = base_select_xgb_best_passo2$min_n,
  mtry = base_select_xgb_best_passo4$mtry,
  trees = tune(),
  tree_depth = base_select_xgb_best_passo2$tree_depth,
  learn_rate = tune(),
  loss_reduction = base_select_xgb_best_passo3$loss_reduction,
  sample_size = base_select_xgb_best_passo4$sample_size
) |>
  set_mode("classification") |>
  set_engine("xgboost", count = FALSE)

## Workflow

base_xb_wf <- workflow() |>
  add_model(base_xb_model) |>
  add_recipe(base_xb_recipe)

## Tune grid para learn rate e trees

grid_xb <- expand.grid(
  learn_rate = c(0.05),
  trees = c(100, 600, 750)
)

grid_xb

# doParallel::registerDoParallel(4)

base_xb_tune_grid <- tune_grid(
  base_xb_wf,
  resamples = base_resamples,
  grid = grid_xb,
  metrics = metric_set(roc_auc)
)

#### Melhores hiperparâmetros

autoplot(base_xb_tune_grid)
show_best(base_xb_tune_grid, n = 6)

base_select_xgb_best_passo5 <- base_xb_tune_grid |>
  select_best(metric = "roc_auc")

base_select_xgb_best_passo5

## Desempenho do Modelo Final

base_xb_model <- boost_tree(
  min_n = base_select_xgb_best_passo2$min_n,
  mtry = base_select_xgb_best_passo4$mtry,
  trees = base_select_xgb_best_passo5$trees,
  tree_depth = base_select_xgb_best_passo2$tree_depth,
  learn_rate = base_select_xgb_best_passo5$learn_rate,
  loss_reduction = base_select_xgb_best_passo3$loss_reduction,
  sample_size = base_select_xgb_best_passo4$sample_size
) |>
  set_mode("classification") |>
  set_engine("xgboost", count = FALSE)

## Workflow

base_xb_wf <- workflow() |>
  add_model(base_xb_model) |>
  add_recipe(base_xb_recipe)

base_last_fit <- base_xb_wf |>
  last_fit(
    split = base_initial_split,
    control = control_grid(save_pred = TRUE, verbose = FALSE, allow_par = TRUE),
    metrics = metric_set(roc_auc, f_meas, accuracy, precision, recall)
  )

#### Métricas
collect_metrics(base_last_fit)

#### Variáveis Importantes

base_last_fit |>
  pluck(".workflow", 1) |>
  pull_workflow_fit() |>
  vip::vip(num_features = 20)

# Desempenho dos modelos finais ----------------------------------------------

base_lr_best_params <- select_best(base_lr_tune_grid, "roc_auc")
base_lr_wf <- base_lr_wf |> finalize_workflow(base_lr_best_params)
base_lr_last_fit <- last_fit(base_lr_wf, base_initial_split)

base_dt_best_params <- select_best(base_dt_tune_grid, "roc_auc")
base_dt_wf <- base_dt_wf |>  finalize_workflow(base_dt_best_params)
base_dt_last_fit <- last_fit(base_dt_wf, base_initial_split)

base_rf_best_params <- select_best(base_rf_tune_grid, "roc_auc")
base_rf_wf <- base_rf_wf |>  finalize_workflow(base_rf_best_params)
base_rf_last_fit <- last_fit(base_rf_wf, base_initial_split)

base_xb_best_params <- select_best(base_xb_tune_grid, "roc_auc")
base_xb_wf <- base_xb_wf |>  finalize_workflow(base_xb_best_params)
base_xb_last_fit <- last_fit(base_xb_wf, base_initial_split)

base_test_preds <- bind_rows(
  collect_predictions(base_lr_last_fit) |> mutate(modelo = "LogisticRegression"),
  collect_predictions(base_dt_last_fit) |> mutate(modelo = "DecisionTree"),
  collect_predictions(base_rf_last_fit) |> mutate(modelo = "RandomForest"),
  collect_predictions(base_xb_last_fit) |> mutate(modelo = "XGBoost")
)

## roc
base_test_preds |>
  group_by(modelo) |>
  roc_curve(heart_disease, .pred_0) |>
  autoplot()

## lift
base_test_preds %>%
  group_by(modelo) %>%
  lift_curve(heart_disease, .pred_0) %>%
  autoplot()

# Variáveis importantes
base_lr_last_fit_model <- base_lr_last_fit$.workflow[[1]]$fit$fit
vip(base_lr_last_fit_model)

base_dt_last_fit_model <- base_dt_last_fit$.workflow[[1]]$fit$fit
vip(base_dt_last_fit_model)

# base_rf_last_fit_model <- base_rf_last_fit$.workflow[[1]]$fit$fit
# vip(base_rf_last_fit_model)

# Guardar tudo ------------------------------------------------------------

write_rds(base_lr_last_fit, "base_lr_last_fit.rds")
write_rds(base_lr_model, "base_lr_model.rds")


# Testando o Modelo final -------------------------------------------------

base_final_xb_model <- base_lr_wf |> fit(base)

test1 <- base[4,]

predict(base_final_lr_model, new_data = base)[4,]





