from utils import print_header
import train_eval


def main():
    print_header("PHASE 4: TRAINING + EVALUATION")
    train_eval.main()
    print_header("PHASE 4 COMPLETE")


if __name__ == "__main__":
    main()