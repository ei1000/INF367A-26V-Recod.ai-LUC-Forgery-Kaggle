from __future__ import annotations


def set_optimizer_learning_rate(optimizer, learning_rate: float):
    if optimizer is None:
        return

    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def set_optimizer_group_learning_rate(optimizer, group_name: str, learning_rate: float):
    if optimizer is None:
        return

    for param_group in optimizer.param_groups:
        if param_group.get("name") == group_name:
            param_group["lr"] = learning_rate


def collect_backbone_parameter_groups(
    pyramid_bb,
    feature_backbone: str,
    cnn_backbone: str,
    learning_rate: float,
    feature_backbone_learning_rate: float | None,
):
    if pyramid_bb is None:
        return []

    backbone_lr = learning_rate if feature_backbone_learning_rate is None else feature_backbone_learning_rate
    parameter_groups = []

    if feature_backbone == "dino" and hasattr(pyramid_bb, "backbone"):
        dino_backbone = pyramid_bb.backbone
        head_params = list(dino_backbone.proj.parameters()) if getattr(dino_backbone, "proj", None) is not None else []
        encoder_params = [
            param
            for name, param in dino_backbone.named_parameters()
            if param.requires_grad and not name.startswith("proj.")
        ]
        if encoder_params:
            parameter_groups.append({"params": encoder_params, "lr": backbone_lr, "name": "feature_backbone"})
        if head_params:
            parameter_groups.append({"params": head_params, "lr": learning_rate, "name": "feature_head"})
        return parameter_groups

    if feature_backbone == "cnn" and cnn_backbone == "pretrained" and hasattr(pyramid_bb, "backbone"):
        cnn_feature_model = pyramid_bb.backbone
        feature_params = list(cnn_feature_model.features.parameters()) if hasattr(cnn_feature_model, "features") else []
        proj_params = list(cnn_feature_model.proj.parameters()) if hasattr(cnn_feature_model, "proj") else []
        if feature_params:
            parameter_groups.append({"params": feature_params, "lr": backbone_lr, "name": "feature_backbone"})
        if proj_params:
            parameter_groups.append({"params": proj_params, "lr": learning_rate, "name": "feature_head"})
        return parameter_groups

    params = [param for param in pyramid_bb.parameters() if param.requires_grad]
    if params:
        parameter_groups.append({"params": params, "lr": learning_rate, "name": "feature_backbone"})
    return parameter_groups
