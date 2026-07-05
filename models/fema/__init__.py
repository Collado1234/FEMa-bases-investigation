from models.registry import register_model
from models.fema.fema_plugin import FEMaPlugin

register_model("fema", FEMaPlugin)
