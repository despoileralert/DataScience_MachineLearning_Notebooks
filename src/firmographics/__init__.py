"""SDS datathon project: cleaning and segmenting a firmographic dataset of
~8.5k (mostly Chinese) companies.

Pipeline: drop redundant columns -> de-duplicate -> handle each kind of
missingness -> separate multivariate outliers -> engineer features ->
cluster (K-Prototypes, K-Means, HDBSCAN, Gower hierarchical) -> profile the
clusters and measure which ones attract the outliers ("anomaly lift").
"""
