"""
visualize.py

Utility to visualize before and after images side-by-side
and save publication-quality figures.
run this file by using the command in terminal: python path/to/visualize.py
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def _read_image(image):
    """
    Reads an image from a file path or returns the numpy array.
    """

    if isinstance(image, str):
        image = np.array(Image.open(image).convert("RGB"))

    return image


def show_comparison(
    before,
    after,
    before_title="Before",
    after_title="After",
    figure_title="Comparison",
    save_path=r"../generated-results/comparison.png",
    dpi=300,
    figsize=(12, 6),
    show=True,
):
    """
    Display one original image beside one or more processed images.
    """

    before = _read_image(before)

    # Allow single image or list
    if not isinstance(after, (list, tuple)):
        after = [after]

    after = [_read_image(img) for img in after]

    n = len(after)

    fig = plt.figure(
        figsize=(figsize[0], max(figsize[1], 3 * n)),
        constrained_layout=True,
    )

    gs = fig.add_gridspec(n, 2)

    # Original image spans all rows
    ax_before = fig.add_subplot(gs[:, 0])
    ax_before.imshow(before)
    ax_before.set_title(before_title, fontsize=14)
    ax_before.axis("off")

    # Processed images
    for i, img in enumerate(after):
        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(img)

        if n == 1:
            title = after_title
        else:
            title = f"{after_title} {i+1}"

        ax.set_title(title, fontsize=14)
        ax.axis("off")

    fig.suptitle(
        figure_title,
        fontsize=18,
        fontweight="bold",
    )

    plt.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.15,
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"✓ Figure saved to: {save_path}")
    print(f"✓ DPI: {dpi}")
    

if __name__ == "__main__":
    # show_comparison(
    # before=r"../sample-inputs/2024-12-24_10_10_40_(18130)_sardine_input.jpeg",
    # after=[r"../generated-results/2024-12-24_10_10_40_(18130)_sardine_input_segmented_0.png",
    #        r"../generated-results/2024-12-24_10_10_40_(18130)_sardine_input_segmented_1.png"],
    # before_title="Original Image",
    # after_title="Segmented Fish",
    # figure_title="Fish Segmentation",
    # save_path=r"../generated-results/comparison1.png",
    # dpi=1200,
    # )
    
    # show_comparison(
    #     before=r"../generated-results/2024-12-24_10_10_40_(18130)_sardine_input_segmented_1.png",
    #     after=r"../generated-results/sardine_fish_cuts.png",
    #     before_title="Segmented Fish",
    #     after_title="Detected Damages",
    #     figure_title="Fish Damage Detection",
    #     save_path=r"../generated-results/comparison2.png",
    #     dpi=600,
    # )
    
    show_comparison(
        before=r"../generated-results/2025-01-28_10_50_33_(19200)_mackerel_input_segmented_0.png",
        after=r"../generated-results/mackerel_fish_cuts.png",
        before_title="Segmented Fish",
        after_title="Detected Damages",
        figure_title="Fish Damage Detection",
        save_path=r"../generated-results/comparison3.png",
        dpi=1200,
    )