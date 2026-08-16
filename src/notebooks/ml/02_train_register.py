# Databricks notebook source
# MAGIC %md
# MAGIC # ML 02 · Train, register, gate
# MAGIC
# MAGIC Trains a challenger on the feature table, logs everything to MLflow,
# MAGIC registers it in Unity Catalog, and only flips the `@champion` alias if the
# MAGIC challenger beats the incumbent on the **same fresh holdout window** — both
# MAGIC models scored on months neither trained on, so the incumbent is never
# MAGIC penalised for having trained on older data.
# MAGIC
# MAGIC Deploy-code-not-models: this notebook *is* the deployable. Rerunning it
# MAGIC retrains from current data; promotion is a decision the gate makes, not a
# MAGIC human copying files.

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from lib.ml_features import FEATURE_COLUMNS, TARGET, split_by_month  # noqa: E402
from lib.ml_model import (  # noqa: E402
    challenger_wins,
    evaluate,
    fit_model,
    naive_baseline_mae,
)

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("ml_schema", "propertyiq_ml")
dbutils.widgets.text("holdout_months", "6")

catalog = dbutils.widgets.get("catalog")
ml_schema = dbutils.widgets.get("ml_schema")
holdout_months = int(dbutils.widgets.get("holdout_months"))

MODEL_NAME = f"{catalog}.{ml_schema}.rent_estimator"

# Registry lives in Unity Catalog; the experiment needs an explicit path when
# running as a job (there is no notebook-default experiment in a job context).
mlflow.set_registry_uri("databricks-uc")
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/propertyiq-rent-estimator")

# COMMAND ----------

features = spark.table(f"{catalog}.{ml_schema}.features_rent")
train_df, test_df = split_by_month(features, holdout_months)

train_pdf = train_df.toPandas()
test_pdf = test_df.toPandas()
print(f"train: {len(train_pdf):,} rows · holdout: {len(test_pdf):,} rows ({holdout_months} months)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train the challenger and log the run
# MAGIC
# MAGIC The naive baseline (next month = last month) is logged alongside: a model
# MAGIC that can't beat it has learned nothing worth deploying, whatever its MAE.

# COMMAND ----------

with mlflow.start_run() as run:
    model = fit_model(train_pdf)
    metrics = evaluate(model, test_pdf)
    baseline = naive_baseline_mae(test_pdf)

    mlflow.log_params(
        {
            "holdout_months": holdout_months,
            "n_train_rows": len(train_pdf),
            "n_test_rows": len(test_pdf),
            "features": ",".join(FEATURE_COLUMNS),
        }
    )
    mlflow.log_metrics({**metrics, "naive_mae": baseline})

    signature = mlflow.models.infer_signature(
        train_pdf[FEATURE_COLUMNS], model.predict(train_pdf[FEATURE_COLUMNS].head())
    )
    logged = mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        input_example=train_pdf[FEATURE_COLUMNS].head(5),
        registered_model_name=MODEL_NAME,
    )

challenger_mae = metrics["mae"]
print(f"challenger MAE ${challenger_mae:.2f}/wk · naive ${baseline:.2f}/wk · R² {metrics['r2']:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### The promotion gate
# MAGIC
# MAGIC Load the current `@champion` (if any) and score it on the *same* holdout
# MAGIC months. The alias only moves on a clear win (2% MAE margin — see
# MAGIC `lib/ml_model.py`, where the rule is unit-tested). Losing challengers stay
# MAGIC registered and tagged, so the version history *is* the audit trail.

# COMMAND ----------

client = MlflowClient()
challenger_version = logged.registered_model_version

try:
    champion_version = client.get_model_version_by_alias(MODEL_NAME, "champion").version
    champion = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
    champion_pred = champion.predict(test_pdf[FEATURE_COLUMNS])
    from sklearn.metrics import mean_absolute_error  # noqa: E402

    champion_mae = float(mean_absolute_error(test_pdf[TARGET], champion_pred))
except Exception:
    champion_version, champion_mae = None, None

promoted = challenger_wins(challenger_mae, champion_mae)

client.set_model_version_tag(MODEL_NAME, challenger_version, "holdout_mae", f"{challenger_mae:.2f}")
client.set_model_version_tag(MODEL_NAME, challenger_version, "naive_mae", f"{baseline:.2f}")

if promoted:
    client.set_registered_model_alias(MODEL_NAME, "champion", challenger_version)
    verdict = f"PROMOTED: v{challenger_version} takes @champion"
    if champion_mae is not None:
        verdict += f" (${challenger_mae:.2f} vs ${champion_mae:.2f})"
else:
    client.set_registered_model_alias(MODEL_NAME, "challenger", challenger_version)
    verdict = (
        f"CHAMPION STANDS: v{champion_version} (${champion_mae:.2f}) holds off "
        f"v{challenger_version} (${challenger_mae:.2f})"
    )

print(verdict)

# Downstream tasks (and the jobs UI) get the verdict as a task value.
dbutils.jobs.taskValues.set("promoted", promoted)
dbutils.jobs.taskValues.set("challenger_mae", challenger_mae)
