# RecEraserPro

This is my implementation of RecEraser and its plus version using contrastive learning.

# Hype-Parameters

The hype-parameters for base models are:

```
yelp2018:
BPR: adagrade	lr=0.05	reg=0.01	batch=256
WMF: adagrade	lr=0.05	reg=0.01	batch=256	weight=0.05	drop=0.7
LightGCN: adam	lr=0.001	reg=1e-4	batch=1024

ml-1m:
BPR: adagrade	lr=0.05	reg=0.01	batch=256
WMF: adagrade	lr=0.05	reg=0.01	batch=256	weight=0.2	drop=0.7
LightGCN: adam lr=0.001	reg=1e-3	batch=1024

ml-10m:

BPR: adagrade	lr=0.05	reg=0.001	batch=256
WMF: adagrade	lr=0.05	reg=0.01	batch=256	weight=0.2	drop=0.7
LightGCN: adam lr=0.001	reg=1e-3	batch=1024
```
