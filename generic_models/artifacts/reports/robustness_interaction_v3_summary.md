# Robustness Evaluation

- Evaluated model: `interaction_v3`
- Sessions: `800`
- Sites: `4`
- Label counts: `{'human': 440, 'bot': 360}`
- This is a local robustness benchmark over varied synthetic humans and bot families; it does not replace real human telemetry collection.

## Overall Metrics

```text
    model_name  threshold  accuracy  precision  recall       f1  false_positive_rate  false_negative_rate  roc_auc  true_negative  false_positive  false_negative  true_positive
interaction_v3       0.05   0.99875    0.99723     1.0 0.998613             0.002273                  0.0      1.0            439               1               0            360
```

## Scenario Results

```text
                 scenario label  sessions  detection_rate  mean_score  errors                                                                                    notes
              bfs_crawler   bot        40           1.000    0.999937       0                                                          Breadth-first systematic crawl.
          bot_py_like_bot   bot        40           1.000    0.999933       0 Real-browser direct goto style with random mouse/scroll noise and no causal click chain.
  browser_mouse_noise_bot   bot        40           1.000    0.999935       0                          Browser-like mouse and scroll noise, but no causal click chain.
        bursty_timing_bot   bot        40           1.000    0.999937       0                                                 Alternates very fast bursts with pauses.
          click_based_bot   bot        40           1.000    0.999910       0                           Uses matching clicks but follows systematic coverage behavior.
      coverage_greedy_bot   bot        40           1.000    0.999937       0                                                   Chooses unvisited/high-coverage pages.
              dfs_crawler   bot        40           1.000    0.999937       0                                                            Depth-first systematic crawl.
          direct_goto_bot   bot        40           1.000    0.999937       0                                Direct navigation to valid URLs without click precursors.
          random_walk_bot   bot        40           1.000    0.999937       0                                                               Graph-valid random walker.
               human_fast human        40           0.000    0.000077       0                                                   Fast human with real click precursors.
human_low_activity_clicks human        40           0.000    0.000090       0                               Sparse telemetry but still causal click-driven navigation.
         human_low_scroll human        40           0.000    0.000051       0                                              Almost no scrolling but still click-driven.
             human_normal human        40           0.000    0.000051       0                                       Task-like browsing with clicks and moderate dwell.
 human_product_then_leave human        40           0.000    0.000052       0                                                    Short product/detail visit then exit.
      human_revisit_heavy human        40           0.000    0.000051       0                                          Human repeatedly backtracks and revisits pages.
       human_scroll_heavy human        40           0.000    0.000051       0                                             Lots of reading/scrolling before navigation.
human_search_cart_contact human        40           0.000    0.000051       0                                                   Uses utility/cart/contact-style paths.
              human_short human        40           0.025    0.025035       1                                         Only 3-4 pages, a key early-detection edge case.
               human_slow human        40           0.000    0.000067       0                                                     Slow human with long reading pauses.
    human_unusual_pattern human        40           0.000    0.000051       0                                          Weird but still causal click-driven human path.
```

## Example Session Table

```text
             session_type expected_label detected_as_bot  score                                                                                    notes
             human_normal          human              No    0.0                                       Task-like browsing with clicks and moderate dwell.
               human_fast          human              No    0.0                                                   Fast human with real click precursors.
human_low_activity_clicks          human              No    0.0                               Sparse telemetry but still causal click-driven navigation.
      human_revisit_heavy          human              No    0.0                                          Human repeatedly backtracks and revisits pages.
              bfs_crawler            bot             Yes    1.0                                                          Breadth-first systematic crawl.
          random_walk_bot            bot             Yes    1.0                                                               Graph-valid random walker.
  browser_mouse_noise_bot            bot             Yes    1.0                          Browser-like mouse and scroll noise, but no causal click chain.
          bot_py_like_bot            bot             Yes    1.0 Real-browser direct goto style with random mouse/scroll noise and no causal click chain.
          direct_goto_bot            bot             Yes    1.0                                Direct navigation to valid URLs without click precursors.
          click_based_bot            bot             Yes    1.0                           Uses matching clicks but follows systematic coverage behavior.
```

## Ablation Study

Ablations are trained as temporary leave-one-site-out models. No old model bundles are replaced.

```text
              ablation  feature_count  accuracy  precision   recall       f1  false_positive_rate  false_negative_rate  roc_auc  true_negative  false_positive  false_negative  true_positive
           timing_only              6   0.99875   0.997230 1.000000 0.998613             0.002273             0.000000 0.999621            439               1               0            360
     graph_plus_timing             55   0.99750   0.997222 0.997222 0.997222             0.002273             0.002778 0.999047            439               1               1            359
      interaction_only             31   0.99625   0.991736 1.000000 0.995851             0.006818             0.000000 1.000000            437               3               0            360
          all_features             85   0.99625   0.991736 1.000000 0.995851             0.006818             0.000000 1.000000            437               3               0            360
graph_plus_interaction             80   0.99625   0.991736 1.000000 0.995851             0.006818             0.000000 0.999949            437               3               0            360
            graph_only             50   0.87000   0.923841 0.775000 0.842900             0.052273             0.225000 0.955079            417              23              81            279
```

## Top Feature Importance

```text
                      feature  importance_mean  importance_std      family
     meaningful_event_density     1.935685e-02    1.619113e-03 interaction
      telemetry_anomaly_score     5.983270e-03    1.751085e-03 interaction
                 path_entropy     2.775558e-17    4.807407e-17       graph
          inter_hop_time_mean     2.775558e-17    4.807407e-17      timing
               coverage_ratio     0.000000e+00    0.000000e+00       graph
                 unique_nodes     0.000000e+00    0.000000e+00       graph
            unique_node_ratio     0.000000e+00    0.000000e+00       graph
      normalized_path_entropy     0.000000e+00    0.000000e+00       graph
                 revisit_rate     0.000000e+00    0.000000e+00       graph
        session_length_so_far     0.000000e+00    0.000000e+00       graph
            edge_revisit_rate     0.000000e+00    0.000000e+00       graph
normalized_transition_entropy     0.000000e+00    0.000000e+00       graph
           transition_entropy     0.000000e+00    0.000000e+00       graph
                   depth_mean     0.000000e+00    0.000000e+00       graph
             deep_visit_ratio     0.000000e+00    0.000000e+00       graph
            branching_entropy     0.000000e+00    0.000000e+00       graph
                    depth_max     0.000000e+00    0.000000e+00       graph
                    depth_std     0.000000e+00    0.000000e+00       graph
       outbound_coverage_mean     0.000000e+00    0.000000e+00       graph
   deterministic_branch_ratio     0.000000e+00    0.000000e+00       graph
```

## Interpretation

- Graph/timing features are useful for systematic crawlers such as BFS, DFS, coverage-greedy, and bursty bots.
- Interaction provenance is the key extra signal for browser-like direct-navigation bots with mouse noise.
- False positives on humans should remain the primary metric to watch when real telemetry is collected.
- The strongest next evidence would be a later time-split using real human sessions captured from the four visual websites.
