import os
import shutil
import random


def split_data(
    input_dir,
    output_dir,
    split_ratio=(0.8, 0.1, 0.1),
    animation_types=["color", "composition"],
):
    """
    Split the data into train, validation, and test sets.
    """
    # Collect all scene directories with their metadata
    scenes = []
    for animation_dir in os.listdir(input_dir):
        animation_dir_path = os.path.join(input_dir, animation_dir)
        if not os.path.isdir(animation_dir_path):
            continue
        for type_dir in os.listdir(animation_dir_path):
            if type_dir not in animation_types:
                continue
            type_dir_path = os.path.join(animation_dir_path, type_dir)
            if not os.path.isdir(type_dir_path):
                continue
            for scene_dir in os.listdir(type_dir_path):
                scene_dir_path = os.path.join(type_dir_path, scene_dir)
                if not os.path.isdir(scene_dir_path):
                    continue
                scenes.append(
                    {
                        "animation_dir": animation_dir,
                        "type_dir": type_dir,
                        "scene_dir": scene_dir,
                        "scene_path": scene_dir_path,
                    }
                )

    # Shuffle scenes for random distribution
    random.shuffle(scenes)

    # Calculate split indices
    total_scenes = len(scenes)
    train_end = int(total_scenes * split_ratio[0])
    val_end = train_end + int(total_scenes * split_ratio[1])

    # Split scenes
    train_scenes = scenes[:train_end]
    val_scenes = scenes[train_end:val_end]
    test_scenes = scenes[val_end:]

    # Copy scenes to respective directories
    splits = {"train": train_scenes, "val": val_scenes, "test": test_scenes}

    for split_name, split_scenes in splits.items():
        for scene_info in split_scenes:
            # Create output directory structure
            output_animation_dir = os.path.join(
                output_dir, split_name, scene_info["animation_dir"]
            )
            output_type_dir = os.path.join(output_animation_dir, scene_info["type_dir"])
            output_scene_dir = os.path.join(output_type_dir, scene_info["scene_dir"])

            # Create directories if they don't exist
            os.makedirs(output_scene_dir, exist_ok=True)

            # Copy all files from scene directory
            for file_name in os.listdir(scene_info["scene_path"]):
                src_file = os.path.join(scene_info["scene_path"], file_name)
                dst_file = os.path.join(output_scene_dir, file_name)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)

    print(
        f"Split complete: {len(train_scenes)} train, {len(val_scenes)} val, {len(test_scenes)} test scenes"
    )


if __name__ == "__main__":
    split_data(
        input_dir="data",
        output_dir="data_split",
        split_ratio=(0.8, 0.1, 0.1),
        animation_types=["color", "composition"],
    )
