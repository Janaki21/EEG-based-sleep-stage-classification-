from utils import print_header
import preprocess


def main():
    print_header("PHASE 2: PREPROCESSING + EPOCH CREATION")
    metadata = preprocess.build_dataset()
    print_header("PHASE 2 COMPLETE")
    print(metadata)


if __name__ == "__main__":
    main()