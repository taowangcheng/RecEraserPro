import models
import dataloaders

MODELS = {
    'LightGCN': models.LightGCN,
    'RecEraser': models.RecEraser,
}
DATALOADERS = {
    'normal': dataloaders.NormalDataLoader,
    'spilit': dataloaders.SpilitDataLoader,
}
