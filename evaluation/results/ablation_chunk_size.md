### Chunk-size sweep (recursive) vs section (article-aware) baseline

| config | strategy | chunk_size | overlap | n_chunks | median_chars | build_s | encode_ms/chunk | recall@5 | precision@5 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| recursive_200 | recursive | 200 | 40 | 22752 | 138.000 | 58.790 | 0.264 | 0.800 | 0.328 | 0.628 | 0.655 |
| recursive_400 | recursive | 400 | 80 | 9919 | 344 | 48.590 | 0.237 | 0.600 | 0.264 | 0.558 | 0.575 |
| recursive_800 | recursive | 800 | 160 | 5085 | 747 | 51.840 | 0.229 | 0.600 | 0.208 | 0.440 | 0.461 |
| recursive_1600 | recursive | 1600 | 320 | 2497 | 1545 | 42.980 | 0.247 | 0.680 | 0.208 | 0.550 | 0.592 |
| section_baseline | section | — | — | 1335 | 1369 | 44.690 | 0.368 | 0.920 | 0.208 | 0.716 | 0.768 |
