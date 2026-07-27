# pip install scanpy 
import pandas as pd
import scanpy as sc
import hisepy as hp

# Grab data from HISE
meta_path = hp.reader.cache_files(['cf5fdfaa-b643-4379-a908-e125e5a58dfe'])
metadata = pd.read_csv(meta_path)

altra_path = hp.reader.cache_files(['7e90bfe0-03b0-438d-b654-cdfbff994bd6'])
adata = sc.read_h5ad(altra_path) 

# Swap PB to KT and remove trailing numbers to get KIT ID
adata.obs["sample.sampleKitGuid"] = (
    adata.obs["pbmc_sample_id"].str.replace("PB", "KT").str.split("-", expand=True)[0]
)

# Sanity check
kits_from_ad = set(adata.obs["sample.sampleKitGuid"].unique())
kits_from_meta = set(metadata["sample.sampleKitGuid"].unique())
missing = kits_from_ad - kits_from_meta
print(f"Kits in adata not in metadata: {missing}")

# Merge metadata into obs while preserving index
adata.obs = (
    adata.obs.reset_index()
    .merge(
        metadata[["subject.subjectGuid", "sample.sampleKitGuid", "specimens.specimenGuid", "cohort.cohortGuid"]],
        on="sample.sampleKitGuid",
        how="left"
    )
    .set_index(adata.obs.index.name or "")
)

adata.write("altra_with_meta.h5ad")