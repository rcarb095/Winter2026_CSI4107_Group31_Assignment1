from neural_rerank import rerank_and_save, _select_device


if __name__ == "__main__":
    device = _select_device()
    output = rerank_and_save(
        model_name="all-mpnet-base-v2",
        run_tag="MPNet-rerank",
        output_filename="Results_rerank_mpnet.txt",
        device=device,
    )
    print(f"MPNet reranked results saved to {output} (device={device})")
