# Robustness Evaluation

- Evaluated model: `interaction_v2`
- Sessions: `720`
- Sites: `4`
- Label counts: `{'human': 400, 'bot': 320}`
- This is a local robustness benchmark over varied synthetic humans and bot families; it does not replace real human telemetry collection.

## Overall Metrics

```text
    model_name  threshold  accuracy  precision  recall      f1  false_positive_rate  false_negative_rate  roc_auc  true_negative  false_positive  false_negative  true_positive
interaction_v2       0.05  0.943056   0.996441   0.875 0.93178               0.0025                0.125  0.99743            399               1              40            280
```

## Scenario Results

```text
                 scenario label  sessions  detection_rate  mean_score  errors                                                           notes
              bfs_crawler   bot        40           1.000    0.999915       0                                 Breadth-first systematic crawl.
  browser_mouse_noise_bot   bot        40           1.000    0.999929       0 Browser-like mouse and scroll noise, but no causal click chain.
        bursty_timing_bot   bot        40           1.000    0.998957       0                        Alternates very fast bursts with pauses.
          click_based_bot   bot        40           0.000    0.000296      40  Uses matching clicks but follows systematic coverage behavior.
      coverage_greedy_bot   bot        40           1.000    0.999111       0                          Chooses unvisited/high-coverage pages.
              dfs_crawler   bot        40           1.000    0.998537       0                                   Depth-first systematic crawl.
          direct_goto_bot   bot        40           1.000    0.999975       0       Direct navigation to valid URLs without click precursors.
          random_walk_bot   bot        40           1.000    0.999667       0                                      Graph-valid random walker.
               human_fast human        40           0.000    0.000093       0                          Fast human with real click precursors.
         human_low_scroll human        40           0.000    0.000037       0                     Almost no scrolling but still click-driven.
             human_normal human        40           0.000    0.000039       0              Task-like browsing with clicks and moderate dwell.
 human_product_then_leave human        40           0.000    0.000054       0                           Short product/detail visit then exit.
      human_revisit_heavy human        40           0.000    0.000041       0                 Human repeatedly backtracks and revisits pages.
       human_scroll_heavy human        40           0.000    0.000040       0                    Lots of reading/scrolling before navigation.
human_search_cart_contact human        40           0.000    0.000037       0                          Uses utility/cart/contact-style paths.
              human_short human        40           0.025    0.004818       1                Only 3-4 pages, a key early-detection edge case.
               human_slow human        40           0.000    0.000086       0                            Slow human with long reading pauses.
    human_unusual_pattern human        40           0.000    0.000111       0                 Weird but still causal click-driven human path.
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
        click_based_bot            bot              No    0.0  Uses matching clicks but follows systematic coverage behavior.
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
                    telemetry_anomaly_score     3.555566e-02    3.988427e-03 interaction
                 hover_on_target_link_ratio     2.707227e-02    3.495346e-03 interaction
                      click_precursor_ratio     2.248242e-02    3.199062e-03 interaction
                     click_href_match_ratio     2.071582e-02    2.565958e-03 interaction
              page_type_dwell_residual_mean     1.712793e-02    2.270985e-03 interaction
                       transition_human_nll     3.204102e-03    1.351552e-03 interaction
last_interaction_to_navigation_mean_seconds     2.593750e-03    3.421037e-04 interaction
                        uniform_dwell_score     3.720703e-04    9.183323e-05 interaction
                         inter_hop_time_std     1.289063e-04    3.149319e-05      timing
                             entry_pagerank     5.551115e-17    5.551115e-17       graph
                                  depth_std     5.551115e-17    5.551115e-17       graph
                    visited_out_degree_mean     5.551115e-17    5.551115e-17       graph
                    branching_concentration     5.551115e-17    5.551115e-17       graph
                          edge_revisit_rate     2.775558e-17    4.807407e-17       graph
                        inter_hop_time_mean     2.775558e-17    4.807407e-17      timing
                                 prefix_len     0.000000e+00    0.000000e+00       graph
                      session_length_so_far     0.000000e+00    0.000000e+00       graph
                             coverage_ratio     0.000000e+00    0.000000e+00       graph
                   visited_betweenness_mean     0.000000e+00    0.000000e+00       graph
                          branching_entropy     0.000000e+00    0.000000e+00       graph
```

## Interpretation

- Graph/timing features are useful for systematic crawlers such as BFS, DFS, coverage-greedy, and bursty bots.
- Interaction provenance is the key extra signal for browser-like direct-navigation bots with mouse noise.
- False positives on humans should remain the primary metric to watch when real telemetry is collected.
- The strongest next evidence would be a later time-split using real human sessions captured from the four visual websites.
