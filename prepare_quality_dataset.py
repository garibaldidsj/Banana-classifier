from pathlib import Path
import shutil


SOURCE = Path(
    "dataset/banana_ripeness/"
    "Banana Ripeness Classification Dataset"
)

DESTINATION = Path(
    "dataset/quality"
)


CLASS_MAPPING = {
    "ripe": "comestivel",
    "overripe": "comestivel",
    "unripe": "verde",
    "rotten": "podre"
}


SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test"
}


def main():

    print("=" * 60)
    print("PREPARANDO DATASET")
    print("=" * 60)

    if not SOURCE.exists():

        raise FileNotFoundError(
            f"Dataset não encontrado:\n{SOURCE}"
        )


    # Cria as pastas

    for destination_split in SPLITS.values():

        for class_name in set(
            CLASS_MAPPING.values()
        ):

            (
                DESTINATION
                / destination_split
                / class_name
            ).mkdir(
                parents=True,
                exist_ok=True
            )


    # Processa train / valid / test

    for source_split, destination_split in SPLITS.items():

        print(
            f"\nProcessando: {source_split}"
        )

        for original_class, new_class in CLASS_MAPPING.items():

            source = (
                SOURCE
                / source_split
                / original_class
            )

            destination = (
                DESTINATION
                / destination_split
                / new_class
            )


            if not source.exists():

                print(
                    f"[ERRO] Não encontrado: "
                    f"{source}"
                )

                continue


            images = [
                p for p in source.iterdir()
                if p.is_file()
                and p.suffix.lower()
                in {
                    ".jpg",
                    ".jpeg",
                    ".png"
                }
            ]


            print(
                f"  {original_class:10} -> "
                f"{new_class:10}: "
                f"{len(images)}"
            )


            for index, image in enumerate(images):

                # Prefixo evita colisão entre
                # ripe e overripe

                new_name = (
                    f"{original_class}_"
                    f"{index:06d}"
                    f"{image.suffix.lower()}"
                )


                shutil.copy2(
                    image,
                    destination / new_name
                )


    print("\n" + "=" * 60)
    print("CONCLUÍDO")
    print("=" * 60)


    # Mostra resultado

    for split in [
        "train",
        "val",
        "test"
    ]:

        print(f"\n{split.upper()}")

        for class_name in [
            "comestivel",
            "verde",
            "podre"
        ]:

            directory = (
                DESTINATION
                / split
                / class_name
            )


            count = len([
                p for p in directory.iterdir()
                if p.is_file()
            ])


            print(
                f"  {class_name:12}: "
                f"{count}"
            )


if __name__ == "__main__":
    main()