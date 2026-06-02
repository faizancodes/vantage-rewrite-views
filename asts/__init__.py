"""ASTS-Spec microbenchmark package.

Decision gate for AST-Subtree Speculative Decoding: measure tree-sitter
incremental parse latency vs target/draft model forward-pass latency, then
compute the projected end-to-end speedup at typical accepted-token lengths.
"""
