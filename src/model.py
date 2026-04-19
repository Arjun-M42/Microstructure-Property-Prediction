from transformers import ViTConfig, ViTForImageClassification


def get_model(num_labels: int = 1) -> ViTForImageClassification:
    config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
    config.num_labels = num_labels
    config.problem_type = "regression"

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        config=config,
        ignore_mismatched_sizes=True,
    )

    return model
