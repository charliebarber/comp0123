import sys
from satgen.post_analysis.network_analysis import network_analysis


def main():
    args = sys.argv[1:]
    if len(args) != 4:
        print("Must supply exactly four arguments")
        print("Usage: python -m satgen.post_analysis.main_analyze_path.py [output_data_dir] [satellite_network_dir] "
              "[dynamic_state_update_interval_ms] [end_time_s]")
        exit(1)
    else:
        network_analysis(
            args[0],
            args[1],
            int(args[2]),
            int(args[3]),
            ""  # Must be executed in satgenpy directory
        )


if __name__ == "__main__":
    main()
