# Robustness Evaluation

- Evaluated model: `interaction_v3`
- Sessions: `720`
- Sites: `4`
- Label counts: `{'human': 400, 'bot': 320}`
- This is a local robustness benchmark over varied synthetic humans and bot families; it does not replace real human telemetry collection.

## Overall Metrics

```text
    model_name  threshold  accuracy  precision  recall  f1  false_positive_rate  false_negative_rate  roc_auc  true_negative  false_positive  false_negative  true_positive
interaction_v3       0.05       1.0        1.0     1.0 1.0                  0.0                  0.0      1.0            400               0               0            320
```

## Scenario Results

```text
                 scenario label  sessions  detection_rate  mean_score  errors                                                           notes
              bfs_crawler   bot        40             1.0    0.999933       0                                 Breadth-first systematic crawl.
  browser_mouse_noise_bot   bot        40             1.0    0.999933       0 Browser-like mouse and scroll noise, but no causal click chain.
        bursty_timing_bot   bot        40             1.0    0.999933       0                        Alternates very fast bursts with pauses.
          click_based_bot   bot        40             1.0    0.999933       0  Uses matching clicks but follows systematic coverage behavior.
      coverage_greedy_bot   bot        40             1.0    0.999933       0                          Chooses unvisited/high-coverage pages.
              dfs_crawler   bot        40             1.0    0.999933       0                                   Depth-first systematic crawl.
          direct_goto_bot   bot        40             1.0    0.999933       0       Direct navigation to valid URLs without click precursors.
          random_walk_bot   bot        40             1.0    0.999933       0                                      Graph-valid random walker.
               human_fast human        40             0.0    0.000053       0                          Fast human with real click precursors.
         human_low_scroll human        40             0.0    0.000053       0                     Almost no scrolling but still click-driven.
             human_normal human        40             0.0    0.000053       0              Task-like browsing with clicks and moderate dwell.
 human_product_then_leave human        40             0.0    0.000053       0                           Short product/detail visit then exit.
      human_revisit_heavy human        40             0.0    0.000053       0                 Human repeatedly backtracks and revisits pages.
       human_scroll_heavy human        40             0.0    0.000053       0                    Lots of reading/scrolling before navigation.
human_search_cart_contact human        40             0.0    0.000053       0                          Uses utility/cart/contact-style paths.
              human_short human        40             0.0    0.000053       0                Only 3-4 pages, a key early-detection edge case.
               human_slow human        40             0.0    0.000053       0                            Slow human with long reading pauses.
    human_unusual_pattern human        40             0.0    0.000053       0                 Weird but still causal click-driven human path.
```

## Example Session Table

```text
           session_type expected_label detected_as_bot  score                                                           notes
           human_normal          human              No    0.0              Task-like browsing with clicks and moderate dwell.
             human_fast          human              No    0.0                          Fast human with real click precursors.
    human_revisit_heavy          human              No    0.0                 Human repeatedly backtracks and revisits pages.
            bfs_crawler            bot             Yes    1.0                                 Breadth-first systematic crawl.
        random_walk_bot            bot             Yes    1.0                                      Graph-valid random walker.
browser_mouse_noise_bot            bot             Yes    1.0 Browser-like mouse and scroll noise, but no causal click chain.
        direct_goto_bot            bot             Yes    1.0       Direct navigation to valid URLs without click precursors.
        click_based_bot            bot             Yes    1.0  Uses matching clicks but follows systematic coverage behavior.
```

## Ablation Study

Ablations are trained as temporary leave-one-site-out models. No old model bundles are replaced.

```text
              ablation  feature_count  accuracy  precision   recall       f1  false_positive_rate  false_negative_rate  roc_auc  true_negative  false_positive  false_negative  true_positive
      interaction_only             28  1.000000   1.000000 1.000000 1.000000               0.0000             0.000000 1.000000            400               0               0            320
graph_plus_interaction             77  1.000000   1.000000 1.000000 1.000000               0.0000             0.000000 1.000000            400               0               0            320
          all_features             82  1.000000   1.000000 1.000000 1.000000               0.0000             0.000000 1.000000            400               0               0            320
     graph_plus_timing             55  0.994444   0.987654 1.000000 0.993789               0.0100             0.000000 0.999484            396               4               0            320
           timing_only              6  0.994444   0.987654 1.000000 0.993789               0.0100             0.000000 0.999469            396               4               0            320
            graph_only             50  0.947222   0.951923 0.928125 0.939873               0.0375             0.071875 0.987359            385              15              23            297
```

## Top Feature Importance

```text
                      feature  importance_mean  importance_std      family
 telemetry_events_per_request         0.505547        0.013543 interaction
        session_length_so_far         0.000000        0.000000       graph
                 path_entropy         0.000000        0.000000       graph
      normalized_path_entropy         0.000000        0.000000       graph
                 revisit_rate         0.000000        0.000000       graph
                 unique_nodes         0.000000        0.000000       graph
            unique_node_ratio         0.000000        0.000000       graph
           transition_entropy         0.000000        0.000000       graph
normalized_transition_entropy         0.000000        0.000000       graph
            edge_revisit_rate         0.000000        0.000000       graph
                   depth_mean         0.000000        0.000000       graph
                    depth_std         0.000000        0.000000       graph
                    depth_max         0.000000        0.000000       graph
   depth_distribution_entropy         0.000000        0.000000       graph
             deep_visit_ratio         0.000000        0.000000       graph
            branching_entropy         0.000000        0.000000       graph
      branching_concentration         0.000000        0.000000       graph
               coverage_ratio         0.000000        0.000000       graph
       outbound_coverage_mean         0.000000        0.000000       graph
   deterministic_branch_ratio         0.000000        0.000000       graph
```

## Interpretation

- Graph/timing features are useful for systematic crawlers such as BFS, DFS, coverage-greedy, and bursty bots.
- Interaction provenance is the key extra signal for browser-like direct-navigation bots with mouse noise.
- False positives on humans should remain the primary metric to watch when real telemetry is collected.
- The strongest next evidence would be a later time-split using real human sessions captured from the four visual websites.
