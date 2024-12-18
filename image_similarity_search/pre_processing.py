import argparse
import logging as LOGGER
import re
from pathlib import Path

import pandas as pd
import tqdm


def extract_face(img):
    """Extract the face from the image name"""
    match = re.search(r"[_|-]([A-Z])(?=[._])", img.stem)
    if not match:
        LOGGER.warning(f"Could not extract face from {img}")
    return match.group(1) if match else img


def join_mappin(df: pd.DataFrame, mapping: pd.DataFrame):
    return pd.merge(df, mapping, on=["BRAND", "product_top_category", "product_type"])


# Function to preprocess the dataset
def process(input: str | Path, mapping: str | Path, output: str | Path):
    # check for the folder and file existence
    input = Path(input) if isinstance(input, str) else input
    output = Path(output) if isinstance(output, str) else output
    mapping = Path(mapping) if isinstance(mapping, str) else mapping

    # Load the descriptions of the different brands from the relative excel files
    ax_desc = pd.read_excel(input / "AX_descriptions.xlsx", sheet_name="EN")
    ea_desc = pd.read_excel(input / "EA_descriptions.xlsx", sheet_name="en")
    ga_ea7_desc = pd.read_excel(input / "GA_EA7_descriptions.xlsx", sheet_name="en")
    face_mapping = pd.read_csv(mapping)

    dst = output.parent / "images"
    assert input.is_dir(), f"Input folder {input} does not exist"
    assert dst.parent.is_dir(), f"Output folder {output} does not exist"
    # extract the images from the zip file
    # print(f"Extracting images from {input / 'images.zip'} to {dst / 'images'}")
    # with zipfile.ZipFile(input / "images.zip", "r") as zip_ref:
    #     zip_ref.extractall(dst)

    # print(f"Done extracting images from {input / 'images.zip'} to {dst / 'images'}")

    # check for the csv file existence
    df_path = input / "inventory.csv"
    assert df_path.exists(), f"File {df_path} does not exist"
    assert mapping.exists(), f"File {mapping} does not exist"

    # logging.info(f"Joining the mapping file {mapping} with the dataset {df_path}")

    df = pd.read_csv(input / "inventory.csv")
    # keep only the necessary columns
    cols_to_keep = [
        "MFC",
        "MODEL",
        "FABRIC",
        "COLOUR",
        "BRAND",
        "item_age_range_category",
        "composition",
        "product_gender_unified",
        "product_top_category",
        "product_type",
        "product_subtyped",
    ]
    df = df.drop(columns=[col for col in df.columns if col not in cols_to_keep])

    # creation of the IMAGE_PATH column
    df["IMAGE_PATH"] = None
    df["DESCRIPTION"] = None

    # definition of the base path
    base_path = dst.joinpath("images", "ecommerce")

    subfolders = [
        "FW2018",
        "FW2019",
        "FW2020",
        "FW2021",
        "FW2022",
        "FW2023",
        "FW2024",
        "SS2018",
        "SS2019",
        "SS2020",
        "SS2021",
        "SS2022",
        "SS2023",
        "SS2024",
        "SS2025",
    ]

    # Funzione per trovare l'immagine corretta in una sottocartella o nella cartella di backup
    def find_filtered_image_path(
        model, fabric, colour, brand, top_category, product_type
    ):
        # Cerca l'immagine nella cartella principale con le sottocartelle specificate se esiste

        dirs = (
            base_path.joinpath(f"{model}_{fabric}_{colour}", subfolder)
            for subfolder in subfolders
        )
        dirs = filter(lambda x: x.is_dir(), dirs)
        for dir in dirs:
            files = [
                f
                for f in dir.iterdir()
                if f.is_file() and f.suffix in [".jpg", ".jpeg", ".png"]
            ]
            face2file = {extract_face(file): file for file in files}
            default_face = face_mapping[
                (face_mapping["BRAND"] == brand)
                & (face_mapping["product_top_category"] == top_category)
                & (face_mapping["product_type"] == product_type)
            ].iloc[0]["face"]
            if default_face in face2file:
                return face2file[default_face].relative_to(dst / "images")
            return files[0].relative_to(dst / "images")
        return None

    # Itera su ogni riga del DataFrame e salva il percorso dell'immagine trovata nella colonna 'IMAGE_PATH'
    for index, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        image_path = find_filtered_image_path(
            row["MODEL"],
            row["FABRIC"],
            row["COLOUR"],
            row["BRAND"],
            row["product_top_category"],
            row["product_type"],
        )
        df.at[index, "IMAGE_PATH"] = image_path
        # add the description present into the excel files to the DESCRIPTION column of the corrispondent BRAND
        if row["BRAND"] == "AX":
            filtered_row = ax_desc[ax_desc["MFC"] == row["MFC"]]
            if not filtered_row.empty:
                df.at[index, "DESCRIPTION"] = filtered_row[
                    "DESCRIZIONE MODELLO"
                ].values[0]
            else:
                df.at[index, "DESCRIPTION"] = None  # Nessuna descrizione trovata

        elif row["BRAND"] == "EA":
            filtered_row = ea_desc[ea_desc["MFC"] == row["MFC"]]
            if not filtered_row.empty:
                df.at[index, "DESCRIPTION"] = filtered_row[
                    "DESCRIZIONE MODELLO"
                ].values[0]
            else:
                df.at[index, "DESCRIPTION"] = None

        elif row["BRAND"] in ["GA", "EA7"]:
            filtered_row = ga_ea7_desc[ga_ea7_desc["MFC"] == row["MFC"]]
            if not filtered_row.empty:
                df.at[index, "DESCRIPTION"] = filtered_row[
                    "Editorial Description"
                ].values[0]
            else:
                df.at[index, "DESCRIPTION"] = None

    # salvataggio del dataset
    df.dropna(subset=["IMAGE_PATH", "DESCRIPTION"], inplace=True)
    df.to_csv(output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Preprocessing",
        description="Stuff to do with the dataset",
    )
    parser.add_argument(
        "--input", "-i", type=str, default="catalogue", help="Input folder"
    )
    parser.add_argument(
        "--mapping",
        "-m",
        type=str,
        default="data/interim/mapping.csv",
        help="Front face mapping file",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="dataset.csv", help="Output file"
    )

    args = parser.parse_args()
    process(args.input, args.mapping, args.output)
