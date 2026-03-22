from utils import print_header
import download_data
import inspect_data

def main():
    print_header("PHASE 0 + PHASE 1")
    download_data.main()
    inspect_data.inspect_first_record()
    print_header("DONE")
    print("Phase 0 and Phase 1 completed successfully.")

if __name__ == "__main__":
    main()
