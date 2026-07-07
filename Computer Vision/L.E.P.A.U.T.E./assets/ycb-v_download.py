from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="bop-benchmark/ycbv",
    allow_patterns=["*.zip", "*.z01"],
    local_dir="./bop_datasets/ycbv",
    repo_type="dataset"
)