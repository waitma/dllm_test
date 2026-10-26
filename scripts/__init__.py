"""Project script package; run entrypoints with Python after activating pllm.

Keep this a regular package so spawned data workers cannot resolve a teacher
repository's unrelated ``scripts`` package instead of this project's modules.
"""
