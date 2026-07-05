from models.registry import register_model
from models.sklearn_logreg.logreg_plugin import LogRegPlugin

register_model("logreg_baseline", LogRegPlugin)
