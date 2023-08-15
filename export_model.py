#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import argparse
import os

import coremltools
import torch

import models
from models.modules.mobileone import reparameterize_model


def parse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--variant", type=str, required=True, help="Provide fastvit model variant name."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Provide location to save exported models.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Provide location of trained checkpoint.",
    )
    return parser


def export(variant: str, output_dir: str, checkpoint: str = None) -> None:
    """Method exports coreml package for mobile inference.

    Args:
        variant: FastViT model variant.
        output_dir: Path to save exported model.
        checkpoint: Path to trained checkpoint. Default: ``None``
    """
    # Create output directory.
    os.makedirs(output_dir, exist_ok=True)

    # Random input tensor for tracing purposes.
    inputs = torch.rand(1, 3, 256, 256)
    inputs_tensor = [
        coremltools.TensorType(
            name="images",
            shape=inputs.shape,
        )
    ]

    # Instantiate model variant.
    model = getattr(models, variant)()
    print(f"Export and Convert Model: {variant}")

    # Always reparameterize before exporting.
    reparameterized_model = reparameterize_model(model)
    if checkpoint is not None:
        print(f"Load checkpoint {checkpoint}")
        chkpt = torch.load(checkpoint)
        reparameterized_model.load_state_dict(chkpt["state_dict"])
    reparameterized_model.eval()

    # Trace and export.
    traced_model = torch.jit.trace(reparameterized_model, torch.Tensor(inputs))
    output_path = os.path.join(output_dir, variant)
    pt_name = output_path + ".pt"
    traced_model.save(pt_name)
    ml_model = coremltools.convert(
        model=pt_name,
        outputs=None,
        inputs=inputs_tensor,
        convert_to="mlprogram",
        debug=False,
    )
    ml_model.save(output_path + ".mlpackage")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to export coreml package file")
    parser = parse_args(parser)
    args = parser.parse_args()

    export(args.variant, args.output_dir, args.checkpoint)
