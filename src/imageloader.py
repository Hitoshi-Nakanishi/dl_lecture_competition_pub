import glob
from pathlib import Path
import pandas as pd
import open_clip
from torch.nn import functional as F
from src.datasets import load_loader

data_dir = Path("/notebooks/DockerShared/tracked/dl_lecture_competition_pub/data")


def load_image_paths(dataset, mode):
    image_paths = pd.read_csv(data_dir / f"{mode}_image_paths.txt", header=None, names=["jpg_path"])
    image_paths = image_paths.assign(image_id=dataset.y.detach().numpy()[None].T)
    image_paths["supclass"] = image_paths["jpg_path"].str.split("/").str[-1].str.slice(0, -8)
    image_paths["jpg_path"] = image_paths["supclass"] + "/" + image_paths["jpg_path"].str.split("/").str[-1]
    return image_paths


def divide_chunks(l, n):
    for i in range(0, len(l), n): 
        yield l[i:i + n]


def load_all_image_paths(filename="path_df.csv"):
    filepath = Path(data_dir / filename)
    if filepath.is_file():
        path_df = pd.read_csv(filepath)
        texts = list(path_df["name"].unique())
        return path_df, texts

    jpg_paths = glob.glob(str(data_dir / "Images/*/*.jpg"))
    path_df = pd.DataFrame(jpg_paths, columns=["jpg_path"])
    path_df["jpg_path"] = path_df["jpg_path"].map(lambda x: "/".join(x.split("/")[-2:]))
    path_df["supclass"] = path_df["jpg_path"].str.split("/").str[-2]
    path_df["name"] = path_df["supclass"].str.replace("_", " ").str.replace("-", " ")
    texts = list(path_df["name"].unique())
    path_df = path_df.merge(pd.DataFrame(texts, columns=["name"]).rename_axis("name_id").reset_index(), how="left")
    path_df = path_df.rename_axis("jpg_id").reset_index()
    path_df.to_csv(filepath)
    return path_df, texts


def encode_image_text_features():
    device = "cuda"
    path_df, texts = load_all_image_paths()
    batch_paths = list(divide_chunks(jpg_paths["jpg_path"], 20))

    vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
        'ViT-H-14', pretrained='laion2b_s32b_b79k', precision='fp32', device=device)
    tokenizer = open_clip.get_tokenizer('ViT-H-14')

    image_features_list = []
    for batch in tqdm(batch_paths):
        image_inputs = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in batch]).to(device)
        with torch.no_grad():
            batch_image_features = vlmodel.encode_image(image_inputs)
        image_features_list.append(batch_image_features)
    image_features = torch.cat(image_features_list, dim=0)
    text_inputs = torch.cat([tokenizer(f"a photo of {t}") for t in texts]).to(device)
    with torch.no_grad():
        text_features = vlmodel.encode_text(text_inputs)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    torch.save({
        "text_features": text_features.cpu(),  # torch.Size([1854, 1024])
        "img_features": image_features.cpu()  # torch.Size([26107, 1024])
    }, data_dir / "precoded_features.pt")

    train_loader = load_loader("train", sg_filter=True)
    val_loader = load_loader("val", sg_filter=True)
    train_image_paths = load_image_paths(train_loader.dataset, mode="train")
    val_image_paths = load_image_paths(val_loader.dataset, mode="val")
    train_image_paths = train_image_paths.merge(path_df, how="left")
    val_image_paths = val_image_paths.merge(path_df, how="left")

    train_image_features = image_features[list(train_image_paths["jpg_id"]), :]
    train_text_features = text_features[list(train_image_paths["name_id"]), :]
    torch.save({
        "text_features": train_text_features.cpu(),
        "img_features": train_image_features.cpu()
    }, data_dir / "train_precoded_features.pt")

    val_image_features = image_features[list(val_image_paths["jpg_id"]), :]
    val_text_features = text_features[list(val_image_paths["name_id"]), :]
    torch.save({
        "text_features": val_text_features.cpu(),
        "img_features": val_image_features.cpu()
    }, data_dir / "val_precoded_features.pt")


if __name__ == "__main__":
    encode_image_text_features()