# COMP0123 LEO Network Analysis

This code adapts code by [hypatia](https://github.com/snkas/hypatia) to provide network science analysis.

My main contributions are the following:
    - `analysis`
        - Generating analysis with the `comp0123_analysis.py` file
        - Generated data on the network saved in `data/`
        - The figures in `figs/`
    - `satgenpy`
        - Took the module from hypatia to use their graph generation tools found in:
            - `distance_tools/distance_tools.py`
            - `tles/read_tles.py`
            - `post_analysis/graph_tools.py` : adapted the function `construct_graph_with_distances()` to perform much quicker
            