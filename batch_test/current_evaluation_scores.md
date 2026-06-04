# Current WorldEval Scores

Generated: 2026-05-29 14:03:04 CST

Scoring notes:
- Physical is reported as pass rate: `correct / (correct + incorrect)`, ignoring irrelevant and unknown questions.
- 3D consistency, interaction, CLIP interaction, chunk, transition, global, and all scores are normalized to 0-1.
- Partial runs are aggregated from the judge JSON files currently present. Blank metric cells mean there are no judge JSON files for that domain/pipeline yet.
- `Judged / Expected` uses available manifests when present, otherwise the latest score summary item count when present.

## Summary By Model Category
| Category | Model | Judged / Expected | Physical | 3D Cons. | General Interact. | All |
| --- | --- | --- | --- | --- | --- | --- |
| Gaming World Model | Matrix-Game 2.0 [9] | 968/968 | 0.325 | 0.255 | 0.105 | 0.244 |
| Gaming World Model | Hunyuan-GameCraft [16] | 186/186 | 0.798 | 0.334 | 0.396 | 0.499 |
| Gaming World Model | LingBot-World [32] | 657/889 | 0.942 | 0.373 | 0.752 | 0.675 |
| Robotics World Model | Cosmos-Predict-2.5 [2] | 999/999 | 0.906 | 0.399 | 0.723 | 0.664 |
| Robotics World Model | WoW [6] | 832/884 | 0.708 | 0.250 | 0.346 | 0.440 |
| General World Model | Rolling Forcing [19] | 891/997 | 0.873 | 0.321 | 0.636 | 0.598 |
| General World Model | LongLive [39] | 996/997 | 0.863 | 0.363 | 0.516 | 0.570 |
| General World Model | Yume-1.5 [22] | 673/971 | 0.863 | 0.301 | 0.662 | 0.598 |
| General World Model | Hunyuan-WorldPlay [28] | 971/971 | 0.692 | 0.424 | 0.301 | 0.468 |

## Scores By Domain
| Domain | Pipeline | Judged / Expected | Physical | 3D Cons. | General Interact. | CLIP Interact. | All | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gaming | Matrix-Game 2.0 [9] | 394/394 | 0.332 | 0.189 | 0.107 | 0.230 | 0.232 | Complete or all available judged |
| Gaming | Hunyuan-GameCraft [16] | 0/0 | - | - | - | - | - | No judge results |
| Gaming | LingBot-World [32] | 81/313 | 0.884 | 0.366 | 0.800 | 0.315 | 0.680 | Partial, missing 232 |
| Gaming | Cosmos-Predict-2.5 [2] | 400/400 | 0.867 | 0.361 | 0.696 | 0.306 | 0.636 | Complete or all available judged |
| Gaming | WoW [6] | 284/284 | 0.633 | 0.223 | 0.251 | 0.247 | 0.387 | Complete or all available judged |
| Gaming | Rolling Forcing [19] | 400/400 | 0.853 | 0.289 | 0.677 | 0.332 | 0.598 | Complete or all available judged |
| Gaming | LongLive [39] | 400/400 | 0.851 | 0.292 | 0.548 | 0.322 | 0.560 | Complete or all available judged |
| Gaming | Yume-1.5 [22] | 97/395 | 0.813 | 0.352 | 0.682 | 0.291 | 0.624 | Partial, missing 298 |
| Gaming | Hunyuan-WorldPlay [28] | 395/395 | 0.852 | 0.348 | 0.470 | 0.296 | 0.551 | Complete or all available judged |
| Embodied | Matrix-Game 2.0 [9] | 388/388 | 0.364 | 0.338 | 0.125 | 0.252 | 0.283 | Complete or all available judged |
| Embodied | Hunyuan-GameCraft [16] | 0 | - | - | - | - | - | No judge results |
| Embodied | LingBot-World [32] | 390/390 | 0.949 | 0.393 | 0.725 | 0.314 | 0.671 | Complete or all available judged |
| Embodied | Cosmos-Predict-2.5 [2] | 399/399 | 0.937 | 0.479 | 0.734 | 0.321 | 0.699 | Complete or all available judged |
| Embodied | WoW [6] | 348/400 | 0.787 | 0.272 | 0.448 | 0.288 | 0.498 | Partial, missing 52 |
| Embodied | Rolling Forcing [19] | 291/397 | 0.870 | 0.389 | 0.557 | 0.329 | 0.590 | Partial, missing 106 |
| Embodied | LongLive [39] | 396/397 | 0.857 | 0.472 | 0.452 | 0.327 | 0.578 | Partial, missing 1 |
| Embodied | Yume-1.5 [22] | 390/390 | 0.851 | 0.288 | 0.631 | 0.312 | 0.575 | Complete or all available judged |
| Embodied | Hunyuan-WorldPlay [28] | 390/390 | 0.630 | 0.600 | 0.231 | 0.309 | 0.491 | Complete or all available judged |
| General | Matrix-Game 2.0 [9] | 186/186 | 0.216 | 0.220 | 0.062 | 0.222 | 0.185 | Complete or all available judged |
| General | Hunyuan-GameCraft [16] | 186/186 | 0.798 | 0.334 | 0.396 | 0.295 | 0.499 | Complete or all available judged |
| General | LingBot-World [32] | 186/186 | 0.963 | 0.335 | 0.790 | 0.311 | 0.680 | Complete or all available judged |
| General | Cosmos-Predict-2.5 [2] | 200/200 | 0.939 | 0.317 | 0.755 | 0.313 | 0.651 | Complete or all available judged |
| General | WoW [6] | 200/200 | 0.692 | 0.251 | 0.305 | 0.256 | 0.414 | Complete or all available judged |
| General | Rolling Forcing [19] | 200/200 | 0.933 | 0.285 | 0.667 | 0.314 | 0.612 | Complete or all available judged |
| General | LongLive [39] | 200/200 | 0.909 | 0.290 | 0.579 | 0.315 | 0.572 | Complete or all available judged |
| General | Yume-1.5 [22] | 186/186 | 0.925 | 0.302 | 0.715 | 0.302 | 0.633 | Complete or all available judged |
| General | Hunyuan-WorldPlay [28] | 186/186 | 0.389 | 0.219 | 0.088 | 0.235 | 0.242 | Complete or all available judged |

## Physical Dimension Pass Rates
| Domain | Pipeline | Judged / Expected | Overall | Mechanics | Thermotics | Material |
| --- | --- | --- | --- | --- | --- | --- |
| Gaming | Matrix-Game 2.0 [9] | 394/394 | 0.332 | 0.433 | 0.172 | 0.184 |
| Gaming | Hunyuan-GameCraft [16] | 0/0 | - | - | - | - |
| Gaming | LingBot-World [32] | 81/313 | 0.884 | 0.983 | 0.450 | 0.969 |
| Gaming | Cosmos-Predict-2.5 [2] | 400/400 | 0.867 | 0.951 | 0.418 | 0.884 |
| Gaming | WoW [6] | 284/284 | 0.633 | 0.806 | 0.226 | 0.446 |
| Gaming | Rolling Forcing [19] | 400/400 | 0.853 | 0.941 | 0.418 | 0.854 |
| Gaming | LongLive [39] | 400/400 | 0.851 | 0.941 | 0.377 | 0.865 |
| Gaming | Yume-1.5 [22] | 97/395 | 0.813 | 0.942 | 0.365 | 0.902 |
| Gaming | Hunyuan-WorldPlay [28] | 395/395 | 0.852 | 0.944 | 0.426 | 0.843 |
| Embodied | Matrix-Game 2.0 [9] | 388/388 | 0.364 | 0.366 | 0.000 | 0.372 |
| Embodied | Hunyuan-GameCraft [16] | 0 | - | - | - | - |
| Embodied | LingBot-World [32] | 390/390 | 0.949 | 0.961 | 0.000 | 0.957 |
| Embodied | Cosmos-Predict-2.5 [2] | 399/399 | 0.937 | 0.939 | 0.000 | 0.968 |
| Embodied | WoW [6] | 348/400 | 0.787 | 0.798 | 0.111 | 0.788 |
| Embodied | Rolling Forcing [19] | 291/397 | 0.870 | 0.857 | 0.000 | 0.935 |
| Embodied | LongLive [39] | 396/397 | 0.857 | 0.864 | 0.000 | 0.869 |
| Embodied | Yume-1.5 [22] | 390/390 | 0.851 | 0.857 | 0.000 | 0.872 |
| Embodied | Hunyuan-WorldPlay [28] | 390/390 | 0.630 | 0.577 | 0.000 | 0.810 |
| General | Matrix-Game 2.0 [9] | 186/186 | 0.216 | 0.246 | 0.097 | 0.036 |
| General | Hunyuan-GameCraft [16] | 186/186 | 0.798 | 0.833 | 0.484 | 0.786 |
| General | LingBot-World [32] | 186/186 | 0.963 | 1.000 | 0.519 | 1.000 |
| General | Cosmos-Predict-2.5 [2] | 200/200 | 0.939 | 0.977 | 0.613 | 0.875 |
| General | WoW [6] | 200/200 | 0.692 | 0.743 | 0.300 | 0.562 |
| General | Rolling Forcing [19] | 200/200 | 0.933 | 0.968 | 0.581 | 0.938 |
| General | LongLive [39] | 200/200 | 0.909 | 0.952 | 0.581 | 0.812 |
| General | Yume-1.5 [22] | 186/186 | 0.925 | 0.979 | 0.370 | 0.906 |
| General | Hunyuan-WorldPlay [28] | 186/186 | 0.389 | 0.430 | 0.323 | 0.036 |

## Physical Question Pass Rates
| Domain | Pipeline | Judged / Expected | Gravity | Buoyancy | Compression | Impact | Melting | Sublimation | Vaporization | Condensation | Deposition | Freezing | Color Mixing | Solubility | Hardness | Combustibility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gaming | Matrix-Game 2.0 [9] | 394/394 | 0.494 | 0.479 | 0.324 | 0.247 | 0.214 | 0.000 | 0.146 | 0.179 | 0.167 | 0.231 | - | - | 0.168 | 0.222 |
| Gaming | Hunyuan-GameCraft [16] | 0/0 | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| Gaming | LingBot-World [32] | 81/313 | 0.986 | 0.944 | 1.000 | 1.000 | 0.429 | - | 0.462 | 0.417 | 0.500 | 0.500 | - | - | 0.976 | 0.957 |
| Gaming | Cosmos-Predict-2.5 [2] | 400/400 | 0.958 | 0.986 | 0.986 | 0.868 | 0.357 | 0.000 | 0.292 | 0.513 | 0.833 | 0.538 | - | - | 0.901 | 0.843 |
| Gaming | WoW [6] | 284/284 | 0.850 | 0.944 | 0.774 | 0.540 | 0.200 | 0.000 | 0.161 | 0.296 | 0.200 | 0.333 | - | - | 0.486 | 0.355 |
| Gaming | Rolling Forcing [19] | 400/400 | 0.949 | 0.987 | 0.947 | 0.871 | 0.357 | 0.500 | 0.298 | 0.475 | 0.667 | 0.615 | - | - | 0.854 | 0.854 |
| Gaming | LongLive [39] | 400/400 | 0.955 | 0.986 | 0.946 | 0.846 | 0.429 | 0.000 | 0.292 | 0.436 | 0.500 | 0.462 | - | - | 0.875 | 0.843 |
| Gaming | Yume-1.5 [22] | 97/395 | 0.955 | 1.000 | 0.900 | 0.786 | 0.600 | 0.000 | 0.286 | 0.333 | 0.500 | 0.600 | - | - | 0.902 | 0.900 |
| Gaming | Hunyuan-WorldPlay [28] | 395/395 | 0.957 | 0.986 | 0.959 | 0.844 | 0.500 | 0.500 | 0.271 | 0.513 | 0.667 | 0.538 | - | - | 0.870 | 0.780 |
| Embodied | Matrix-Game 2.0 [9] | 388/388 | 0.427 | 0.467 | 0.290 | 0.239 | - | - | 0.000 | - | - | - | - | 0.000 | 0.374 | - |
| Embodied | Hunyuan-GameCraft [16] | 0 | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| Embodied | LingBot-World [32] | 390/390 | 0.965 | 1.000 | 0.985 | 0.935 | - | - | 0.000 | - | - | - | 1.000 | 0.500 | 0.962 | - |
| Embodied | Cosmos-Predict-2.5 [2] | 399/399 | 0.945 | 1.000 | 1.000 | 0.894 | - | - | 0.000 | - | - | - | 1.000 | 0.500 | 0.972 | - |
| Embodied | WoW [6] | 348/400 | 0.840 | 0.929 | 0.845 | 0.662 | - | - | 0.111 | - | - | - | 0.000 | 0.000 | 0.800 | - |
| Embodied | Rolling Forcing [19] | 291/397 | 0.889 | 1.000 | 0.889 | 0.766 | - | - | 0.000 | - | - | - | - | 0.000 | 0.941 | - |
| Embodied | LongLive [39] | 396/397 | 0.879 | 0.933 | 0.938 | 0.791 | - | - | 0.000 | - | - | - | - | 0.000 | 0.873 | - |
| Embodied | Yume-1.5 [22] | 390/390 | 0.874 | 1.000 | 0.938 | 0.765 | - | - | 0.000 | - | - | - | 0.000 | 0.000 | 0.885 | - |
| Embodied | Hunyuan-WorldPlay [28] | 390/390 | 0.644 | 0.812 | 0.615 | 0.373 | - | - | 0.000 | - | - | - | 0.000 | 0.000 | 0.822 | - |
| General | Matrix-Game 2.0 [9] | 186/186 | 0.310 | 0.267 | 0.111 | 0.130 | 0.125 | - | 0.000 | 0.000 | 1.000 | 0.000 | - | - | 0.037 | 0.000 |
| General | Hunyuan-GameCraft [16] | 186/186 | 0.863 | 0.933 | 0.833 | 0.727 | 0.625 | - | 0.267 | 1.000 | 1.000 | 0.600 | - | - | 0.815 | 0.000 |
| General | LingBot-World [32] | 186/186 | 1.000 | 1.000 | 1.000 | 1.000 | 0.600 | - | 0.400 | 0.500 | 1.000 | 0.667 | - | - | 1.000 | 1.000 |
| General | Cosmos-Predict-2.5 [2] | 200/200 | 0.972 | 1.000 | 1.000 | 0.976 | 0.875 | - | 0.333 | 1.000 | 1.000 | 0.800 | - | - | 0.903 | 0.000 |
| General | WoW [6] | 200/200 | 0.767 | 0.806 | 0.889 | 0.634 | 0.500 | - | 0.133 | - | 0.500 | 0.400 | - | - | 0.581 | 0.000 |
| General | Rolling Forcing [19] | 200/200 | 0.978 | 0.935 | 1.000 | 0.951 | 0.875 | - | 0.267 | 1.000 | 1.000 | 0.800 | - | - | 0.935 | 1.000 |
| General | LongLive [39] | 200/200 | 0.956 | 1.000 | 1.000 | 0.915 | 0.875 | - | 0.400 | 0.000 | 0.500 | 0.800 | - | - | 0.839 | 0.000 |
| General | Yume-1.5 [22] | 186/186 | 0.983 | 1.000 | 1.000 | 0.958 | 0.400 | - | 0.267 | 0.500 | 1.000 | 0.333 | - | - | 0.903 | 1.000 |
| General | Hunyuan-WorldPlay [28] | 186/186 | 0.494 | 0.467 | 0.500 | 0.260 | 0.500 | - | 0.067 | 0.000 | 1.000 | 0.600 | - | - | 0.037 | 0.000 |

## 3D Consistency Submetrics
| Domain | Pipeline | Judged / Expected | GS | Meta | Camera Motion | 3D Cons. |
| --- | --- | --- | --- | --- | --- | --- |
| Gaming | Matrix-Game 2.0 [9] | 394/394 | 0.160 | 0.159 | 0.247 | 0.189 |
| Gaming | Hunyuan-GameCraft [16] | 0/0 | - | - | - | - |
| Gaming | LingBot-World [32] | 81/313 | 0.389 | 0.372 | 0.337 | 0.366 |
| Gaming | Cosmos-Predict-2.5 [2] | 400/400 | 0.415 | 0.388 | 0.280 | 0.361 |
| Gaming | WoW [6] | 284/284 | 0.232 | 0.205 | 0.231 | 0.223 |
| Gaming | Rolling Forcing [19] | 400/400 | 0.324 | 0.292 | 0.250 | 0.289 |
| Gaming | LongLive [39] | 400/400 | 0.328 | 0.292 | 0.256 | 0.292 |
| Gaming | Yume-1.5 [22] | 97/395 | 0.381 | 0.361 | 0.315 | 0.352 |
| Gaming | Hunyuan-WorldPlay [28] | 395/395 | 0.397 | 0.363 | 0.284 | 0.348 |
| Embodied | Matrix-Game 2.0 [9] | 388/388 | 0.283 | 0.298 | 0.432 | 0.338 |
| Embodied | Hunyuan-GameCraft [16] | 0 | - | - | - | - |
| Embodied | LingBot-World [32] | 390/390 | 0.416 | 0.416 | 0.348 | 0.393 |
| Embodied | Cosmos-Predict-2.5 [2] | 399/399 | 0.451 | 0.464 | 0.523 | 0.479 |
| Embodied | WoW [6] | 348/400 | 0.297 | 0.289 | 0.232 | 0.272 |
| Embodied | Rolling Forcing [19] | 291/397 | 0.458 | 0.432 | 0.278 | 0.389 |
| Embodied | LongLive [39] | 396/397 | 0.476 | 0.483 | 0.458 | 0.472 |
| Embodied | Yume-1.5 [22] | 390/390 | 0.337 | 0.340 | 0.185 | 0.288 |
| Embodied | Hunyuan-WorldPlay [28] | 390/390 | 0.574 | 0.566 | 0.660 | 0.600 |
| General | Matrix-Game 2.0 [9] | 186/186 | 0.191 | 0.196 | 0.271 | 0.220 |
| General | Hunyuan-GameCraft [16] | 186/186 | 0.380 | 0.366 | 0.258 | 0.334 |
| General | LingBot-World [32] | 186/186 | 0.373 | 0.319 | 0.312 | 0.335 |
| General | Cosmos-Predict-2.5 [2] | 200/200 | 0.341 | 0.322 | 0.288 | 0.317 |
| General | WoW [6] | 200/200 | 0.243 | 0.225 | 0.286 | 0.251 |
| General | Rolling Forcing [19] | 200/200 | 0.283 | 0.266 | 0.306 | 0.285 |
| General | LongLive [39] | 200/200 | 0.289 | 0.280 | 0.300 | 0.290 |
| General | Yume-1.5 [22] | 186/186 | 0.318 | 0.306 | 0.282 | 0.302 |
| General | Hunyuan-WorldPlay [28] | 186/186 | 0.177 | 0.191 | 0.288 | 0.219 |

## Interaction Submetrics
| Domain | Pipeline | Judged / Expected | Chunk | Transition | Global | Long Range | Global Text | CLIP Interact. | General Interact. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gaming | Matrix-Game 2.0 [9] | 394/394 | 0.135 | 0.074 | 0.087 | 0.087 | 0.087 | 0.230 | 0.107 |
| Gaming | Hunyuan-GameCraft [16] | 0/0 | - | - | - | - | - | - | - |
| Gaming | LingBot-World [32] | 81/313 | 0.796 | 0.767 | 0.862 | 0.875 | 0.848 | 0.315 | 0.800 |
| Gaming | Cosmos-Predict-2.5 [2] | 400/400 | 0.704 | 0.677 | 0.700 | 0.718 | 0.680 | 0.306 | 0.696 |
| Gaming | WoW [6] | 284/284 | 0.267 | 0.233 | 0.247 | 0.250 | 0.244 | 0.247 | 0.251 |
| Gaming | Rolling Forcing [19] | 400/400 | 0.665 | 0.681 | 0.704 | 0.733 | 0.675 | 0.332 | 0.677 |
| Gaming | LongLive [39] | 400/400 | 0.595 | 0.444 | 0.625 | 0.640 | 0.606 | 0.322 | 0.548 |
| Gaming | Yume-1.5 [22] | 97/395 | 0.645 | 0.727 | 0.668 | 0.702 | 0.632 | 0.291 | 0.682 |
| Gaming | Hunyuan-WorldPlay [28] | 395/395 | 0.483 | 0.458 | 0.440 | 0.464 | 0.415 | 0.296 | 0.470 |
| Embodied | Matrix-Game 2.0 [9] | 388/388 | 0.136 | 0.167 | 0.041 | 0.042 | 0.041 | 0.252 | 0.125 |
| Embodied | Hunyuan-GameCraft [16] | 0 | - | - | - | - | - | - | - |
| Embodied | LingBot-World [32] | 390/390 | 0.670 | 0.714 | 0.881 | 0.893 | 0.869 | 0.314 | 0.725 |
| Embodied | Cosmos-Predict-2.5 [2] | 399/399 | 0.682 | 0.707 | 0.896 | 0.908 | 0.885 | 0.321 | 0.734 |
| Embodied | WoW [6] | 348/400 | 0.413 | 0.501 | 0.472 | 0.493 | 0.451 | 0.288 | 0.448 |
| Embodied | Rolling Forcing [19] | 291/397 | 0.498 | 0.600 | 0.632 | 0.661 | 0.603 | 0.329 | 0.557 |
| Embodied | LongLive [39] | 396/397 | 0.484 | 0.288 | 0.587 | 0.619 | 0.556 | 0.327 | 0.452 |
| Embodied | Yume-1.5 [22] | 390/390 | 0.553 | 0.715 | 0.694 | 0.722 | 0.667 | 0.312 | 0.631 |
| Embodied | Hunyuan-WorldPlay [28] | 390/390 | 0.245 | 0.254 | 0.134 | 0.140 | 0.129 | 0.309 | 0.231 |
| General | Matrix-Game 2.0 [9] | 186/186 | 0.069 | 0.064 | 0.031 | 0.031 | 0.029 | 0.222 | 0.062 |
| General | Hunyuan-GameCraft [16] | 186/186 | 0.373 | 0.424 | 0.388 | 0.400 | 0.373 | 0.295 | 0.396 |
| General | LingBot-World [32] | 186/186 | 0.752 | 0.819 | 0.829 | 0.838 | 0.812 | 0.311 | 0.790 |
| General | Cosmos-Predict-2.5 [2] | 200/200 | 0.746 | 0.755 | 0.764 | 0.782 | 0.746 | 0.313 | 0.755 |
| General | WoW [6] | 200/200 | 0.314 | 0.294 | 0.292 | 0.293 | 0.285 | 0.256 | 0.305 |
| General | Rolling Forcing [19] | 200/200 | 0.620 | 0.727 | 0.661 | 0.684 | 0.629 | 0.314 | 0.667 |
| General | LongLive [39] | 200/200 | 0.598 | 0.520 | 0.639 | 0.661 | 0.603 | 0.315 | 0.579 |
| General | Yume-1.5 [22] | 186/186 | 0.637 | 0.815 | 0.718 | 0.748 | 0.684 | 0.302 | 0.715 |
| General | Hunyuan-WorldPlay [28] | 186/186 | 0.130 | 0.030 | 0.073 | 0.073 | 0.073 | 0.235 | 0.088 |

## Model-Level 3D Submetrics
| Category | Model | Judged / Expected | GS | Meta | Camera Motion | 3D Cons. |
| --- | --- | --- | --- | --- | --- | --- |
| Gaming World Model | Matrix-Game 2.0 [9] | 968/968 | 0.216 | 0.222 | 0.326 | 0.255 |
| Gaming World Model | Hunyuan-GameCraft [16] | 186/186 | 0.380 | 0.366 | 0.258 | 0.334 |
| Gaming World Model | LingBot-World [32] | 657/889 | 0.400 | 0.383 | 0.337 | 0.373 |
| Robotics World Model | Cosmos-Predict-2.5 [2] | 999/999 | 0.415 | 0.405 | 0.378 | 0.399 |
| Robotics World Model | WoW [6] | 832/884 | 0.262 | 0.245 | 0.244 | 0.250 |
| General World Model | Rolling Forcing [19] | 891/997 | 0.359 | 0.332 | 0.272 | 0.321 |
| General World Model | LongLive [39] | 996/997 | 0.379 | 0.365 | 0.345 | 0.363 |
| General World Model | Yume-1.5 [22] | 673/971 | 0.338 | 0.334 | 0.231 | 0.301 |
| General World Model | Hunyuan-WorldPlay [28] | 971/971 | 0.426 | 0.412 | 0.436 | 0.424 |

## Model-Level Interaction Submetrics
| Category | Model | Judged / Expected | Chunk | Transition | Global | Long Range | Global Text | CLIP Interact. | General Interact. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gaming World Model | Matrix-Game 2.0 [9] | 968/968 | 0.123 | 0.109 | 0.058 | 0.058 | 0.057 | 0.237 | 0.105 |
| Gaming World Model | Hunyuan-GameCraft [16] | 186/186 | 0.373 | 0.424 | 0.388 | 0.400 | 0.373 | 0.295 | 0.396 |
| Gaming World Model | LingBot-World [32] | 657/889 | 0.709 | 0.751 | 0.864 | 0.875 | 0.850 | 0.314 | 0.752 |
| Robotics World Model | Cosmos-Predict-2.5 [2] | 999/999 | 0.704 | 0.705 | 0.791 | 0.807 | 0.775 | 0.313 | 0.723 |
| Robotics World Model | WoW [6] | 832/884 | 0.339 | 0.359 | 0.352 | 0.362 | 0.341 | 0.267 | 0.346 |
| General World Model | Rolling Forcing [19] | 891/997 | 0.600 | 0.666 | 0.671 | 0.699 | 0.641 | 0.327 | 0.636 |
| General World Model | LongLive [39] | 996/997 | 0.552 | 0.398 | 0.613 | 0.636 | 0.585 | 0.323 | 0.516 |
| General World Model | Yume-1.5 [22] | 673/971 | 0.590 | 0.745 | 0.697 | 0.726 | 0.667 | 0.306 | 0.662 |
| General World Model | Hunyuan-WorldPlay [28] | 971/971 | 0.320 | 0.294 | 0.247 | 0.259 | 0.235 | 0.290 | 0.301 |

