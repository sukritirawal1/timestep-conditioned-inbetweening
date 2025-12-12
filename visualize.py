from pathlib import Path
from PIL import Image
import os
import argparse


def make_filmstrip(
    image_dir,
    output_path="filmstrip.png",
    target_height=300,
    between_frame_name="inbetween",
):
    image_paths = [
        os.path.join(image_dir, f"00_start.png"),
        os.path.join(image_dir, f"01_{between_frame_name}.png"),
        os.path.join(image_dir, f"02_{between_frame_name}.png"),
        os.path.join(image_dir, f"03_{between_frame_name}.png"),
        os.path.join(image_dir, f"04_end.png"),
    ]
    if len(image_paths) != 5:
        raise ValueError("Exactly 5 image paths are required.")

    images = [Image.open(p) for p in image_paths]

    # Convert all to RGB to avoid mode issues
    images = [img.convert("RGB") for img in images]

    # Resize all to the same height, preserving aspect ratio
    resized = []
    for img in images:
        w, h = img.size
        new_w = int(w * (target_height / h))
        resized.append(img.resize((new_w, target_height), Image.LANCZOS))

    # Compute total width and create output image
    total_width = sum(img.size[0] for img in resized)
    filmstrip = Image.new("RGB", (total_width, target_height), (0, 0, 0))

    # Paste images side by side
    x_offset = 0
    for img in resized:
        filmstrip.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    filmstrip.save(output_path)
    print(f"Saved film strip to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", "-s", type=str, default="output")
    parser.add_argument("--output_path", "-o", type=str, default="filmstrip.png")
    parser.add_argument("--use_gt", "-g", action="store_true")
    args = parser.parse_args()

    name = "target" if args.use_gt else "inbetween"

    make_filmstrip(args.input_dir, args.output_path, between_frame_name=name)
