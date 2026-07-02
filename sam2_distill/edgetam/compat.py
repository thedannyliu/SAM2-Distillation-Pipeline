"""Compatibility patches for external EdgeTAM/SAM2 training code."""

from __future__ import annotations

import math


def patch_edgetam_perceiver_view() -> None:
    """Patch EdgeTAM PerceiverResampler for expanded multi-object tensors.

    Upstream EdgeTAM uses ``expand(...).view(...)`` in ``forward_2d``. That
    works for a single object but fails for multi-object image batches because
    expanded tensors are not view-compatible. ``reshape`` preserves the intended
    tensor layout and supports the same single-object path.
    """

    from sam2.modeling.perceiver import PerceiverResampler, window_partition

    if getattr(PerceiverResampler, "_sam2_distill_forward_2d_patch", False):
        return

    def forward_2d(self, x):
        batch, channels, height, width = x.shape
        latents_2d = (
            self.latents_2d.unsqueeze(0)
            .expand(batch, -1, -1)
            .reshape(-1, 1, channels)
        )

        num_window = int(math.sqrt(self.num_latents_2d))
        window_size = height // num_window
        x_windows = x.permute(0, 2, 3, 1)
        x_windows = window_partition(x_windows, window_size).flatten(1, 2)

        for layer in self.layers:
            latents_2d = layer(latents_2d, x_windows)

        latents_2d = latents_2d.reshape(batch, num_window, num_window, channels).permute(0, 3, 1, 2)
        pos_2d = self.position_encoding(latents_2d)
        pos_2d = pos_2d.permute(0, 2, 3, 1).flatten(1, 2)
        latents_2d = latents_2d.permute(0, 2, 3, 1).flatten(1, 2)
        return latents_2d, pos_2d

    PerceiverResampler.forward_2d = forward_2d
    PerceiverResampler._sam2_distill_forward_2d_patch = True
