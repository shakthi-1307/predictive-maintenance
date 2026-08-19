import subprocess
import sys


STEPS = [
    [
        sys.executable,
        "-m",
        "src.generate_data",
    ],
    [
        sys.executable,
        "-m",
        "src.validate_data",
    ],
    [
        sys.executable,
        "-m",
        "src.features",
    ],
    [
        sys.executable,
        "-m",
        "src.train",
    ],
]


def main():

    for step in STEPS:

        print("\n")
        print("=" * 70)
        print(
            "RUNNING:",
            " ".join(step),
        )
        print("=" * 70)

        result = subprocess.run(
            step,
            check=False,
        )

        if result.returncode != 0:

            print(
                "\nPipeline failed."
            )

            sys.exit(
                result.returncode
            )

    print("\n")
    print("=" * 70)
    print("ENTIRE ML PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()