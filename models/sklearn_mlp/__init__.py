from models.registry import register_model
from models.sklearn_mlp.mlp_plugin import MLPPlugin

register_model("cnn", MLPPlugin)
