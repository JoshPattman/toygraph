# `toygraph` - A Go Project for Computational Graphs

I created toygraph as a hands-on way to better understand computational graphs and automatic differentiation. While I’m not making any speed guarantees, I’ve focused on performance where I can (e.g., reusing allocated matrices).

toygraph is designed to be flexible with the types of data it can handle. I started with single-value variables and gradually expanded to include matrices and vectors, all while keeping support for scalar values. The nodes in toygraph are typed, ensuring type safety at compile time, though matrix/vector shape safety is only checked at runtime. You can mix scalar and matrix/vector values within the same graph, but be aware that I don’t plan to add tensor support—so this probably isn’t the tool for machine learning.

Currently, toygraph only runs on the CPU, though extending it to GPU support is something I might explore down the line (it would require a lot more code and operations). While the number of operations available is still limited, adding more is pretty straightforward. You can even use your own types, like a different matrix package than gonum, to build a graph.
