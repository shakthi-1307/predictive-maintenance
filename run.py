import subprocess
import sys


STEPS = [
    [
        sys.executable,
        "-m",
        "src.validation",
    ],
    [
        sys.executable,
        "-m",
        "src.preprocessing",
    ],
    [
        sys.executable,
        "-m",
        "src.features",
    ],
    [
        sys.executable,
        "-m",
        "src.train_baseline",
    ],
    [
        sys.executable,
        "-m",
        "src.evaluate_test",
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
