```
Use device: cuda
MLP(
  (model): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.2, inplace=False)
    (3): Linear(in_features=512, out_features=256, bias=True)
    (4): ReLU()
    (5): Dropout(p=0.2, inplace=False)
    (6): Linear(in_features=256, out_features=128, bias=True)
    (7): ReLU()
    (8): Dropout(p=0.2, inplace=False)
    (9): Linear(in_features=128, out_features=10, bias=True)
  )
)
training progress:   5%|▌         | 1/20 [00:13<04:11, 13.24s/it]
epoch [1/20], loss: 0.5530, test accuracy: 83.44%
training progress:  10%|█         | 2/20 [00:26<04:00, 13.35s/it]
epoch [2/20], loss: 0.4166, test accuracy: 84.09%
training progress:  15%|█▌        | 3/20 [00:40<03:47, 13.37s/it]
epoch [3/20], loss: 0.3836, test accuracy: 86.26%
training progress:  20%|██        | 4/20 [00:53<03:33, 13.37s/it]
epoch [4/20], loss: 0.3601, test accuracy: 86.50%
training progress:  25%|██▌       | 5/20 [01:06<03:19, 13.31s/it]
epoch [5/20], loss: 0.3424, test accuracy: 86.88%
training progress:  30%|███       | 6/20 [01:20<03:06, 13.34s/it]
epoch [6/20], loss: 0.3282, test accuracy: 87.76%
training progress:  35%|███▌      | 7/20 [01:33<02:54, 13.41s/it]
epoch [7/20], loss: 0.3185, test accuracy: 87.53%
training progress:  40%|████      | 8/20 [01:46<02:39, 13.31s/it]
epoch [8/20], loss: 0.3061, test accuracy: 88.03%
training progress:  45%|████▌     | 9/20 [01:59<02:25, 13.26s/it]
epoch [9/20], loss: 0.3000, test accuracy: 87.49%
training progress:  50%|█████     | 10/20 [02:13<02:12, 13.30s/it]
epoch [10/20], loss: 0.2919, test accuracy: 88.11%
training progress:  55%|█████▌    | 11/20 [02:26<01:59, 13.30s/it]
epoch [11/20], loss: 0.2832, test accuracy: 88.57%
training progress:  60%|██████    | 12/20 [02:39<01:46, 13.30s/it]
epoch [12/20], loss: 0.2778, test accuracy: 88.11%
training progress:  65%|██████▌   | 13/20 [02:53<01:32, 13.27s/it]
epoch [13/20], loss: 0.2724, test accuracy: 88.31%
training progress:  70%|███████   | 14/20 [03:06<01:19, 13.24s/it]
epoch [14/20], loss: 0.2686, test accuracy: 88.66%
training progress:  75%|███████▌  | 15/20 [03:19<01:06, 13.21s/it]
epoch [15/20], loss: 0.2634, test accuracy: 88.91%
training progress:  80%|████████  | 16/20 [03:32<00:52, 13.16s/it]
epoch [16/20], loss: 0.2566, test accuracy: 88.44%
training progress:  85%|████████▌ | 17/20 [03:45<00:39, 13.20s/it]
epoch [17/20], loss: 0.2504, test accuracy: 88.53%
training progress:  90%|█████████ | 18/20 [03:58<00:26, 13.17s/it]
epoch [18/20], loss: 0.2487, test accuracy: 88.70%
training progress:  95%|█████████▌| 19/20 [04:12<00:13, 13.22s/it]
epoch [19/20], loss: 0.2444, test accuracy: 88.78%
training progress: 100%|██████████| 20/20 [04:25<00:00, 13.26s/it]
epoch [20/20], loss: 0.2431, test accuracy: 89.03%

Training MLP model #1 with hidden layer design: [128]
training progress:   5%|▌         | 1/20 [00:13<04:08, 13.08s/it]
epoch [1/20], loss: 0.5330, test accuracy: 84.33%
training progress:  10%|█         | 2/20 [00:25<03:53, 12.96s/it]
epoch [2/20], loss: 0.4123, test accuracy: 85.09%
training progress:  15%|█▌        | 3/20 [00:38<03:39, 12.90s/it]
epoch [3/20], loss: 0.3756, test accuracy: 86.22%
training progress:  20%|██        | 4/20 [00:51<03:26, 12.88s/it]
epoch [4/20], loss: 0.3541, test accuracy: 87.00%
training progress:  25%|██▌       | 5/20 [01:04<03:12, 12.87s/it]
epoch [5/20], loss: 0.3388, test accuracy: 86.48%
training progress:  30%|███       | 6/20 [01:17<02:59, 12.80s/it]
epoch [6/20], loss: 0.3269, test accuracy: 86.98%
training progress:  35%|███▌      | 7/20 [01:29<02:46, 12.79s/it]
epoch [7/20], loss: 0.3145, test accuracy: 86.46%
training progress:  40%|████      | 8/20 [01:42<02:33, 12.76s/it]
epoch [8/20], loss: 0.3061, test accuracy: 87.98%
training progress:  45%|████▌     | 9/20 [01:55<02:19, 12.72s/it]
epoch [9/20], loss: 0.2972, test accuracy: 87.75%
training progress:  50%|█████     | 10/20 [02:07<02:07, 12.71s/it]
epoch [10/20], loss: 0.2885, test accuracy: 87.86%
training progress:  55%|█████▌    | 11/20 [02:20<01:54, 12.70s/it]
epoch [11/20], loss: 0.2839, test accuracy: 87.49%
training progress:  60%|██████    | 12/20 [02:33<01:41, 12.66s/it]
epoch [12/20], loss: 0.2776, test accuracy: 87.85%
training progress:  65%|██████▌   | 13/20 [02:45<01:28, 12.68s/it]
epoch [13/20], loss: 0.2719, test accuracy: 87.82%
training progress:  70%|███████   | 14/20 [02:58<01:16, 12.71s/it]
epoch [14/20], loss: 0.2686, test accuracy: 87.97%
training progress:  75%|███████▌  | 15/20 [03:11<01:03, 12.73s/it]
epoch [15/20], loss: 0.2632, test accuracy: 88.22%
training progress:  80%|████████  | 16/20 [03:24<00:50, 12.70s/it]
epoch [16/20], loss: 0.2611, test accuracy: 88.07%
training progress:  85%|████████▌ | 17/20 [03:36<00:37, 12.64s/it]
epoch [17/20], loss: 0.2526, test accuracy: 88.02%
training progress:  90%|█████████ | 18/20 [03:49<00:25, 12.67s/it]
epoch [18/20], loss: 0.2514, test accuracy: 87.87%
training progress:  95%|█████████▌| 19/20 [04:02<00:12, 12.67s/it]
epoch [19/20], loss: 0.2445, test accuracy: 88.29%
training progress: 100%|██████████| 20/20 [04:14<00:00, 12.74s/it]
epoch [20/20], loss: 0.2424, test accuracy: 87.90%

Training MLP model #2 with hidden layer design: [256, 128]
training progress:   5%|▌         | 1/20 [00:12<04:04, 12.87s/it]
epoch [1/20], loss: 0.5406, test accuracy: 84.19%
training progress:  10%|█         | 2/20 [00:25<03:51, 12.89s/it]
epoch [2/20], loss: 0.4067, test accuracy: 85.16%
training progress:  15%|█▌        | 3/20 [00:38<03:41, 13.01s/it]
epoch [3/20], loss: 0.3733, test accuracy: 86.32%
training progress:  20%|██        | 4/20 [00:51<03:28, 13.01s/it]
epoch [4/20], loss: 0.3509, test accuracy: 86.94%
training progress:  25%|██▌       | 5/20 [01:04<03:15, 13.01s/it]
epoch [5/20], loss: 0.3365, test accuracy: 87.41%
training progress:  30%|███       | 6/20 [01:17<03:01, 12.96s/it]
epoch [6/20], loss: 0.3228, test accuracy: 87.25%
training progress:  35%|███▌      | 7/20 [01:30<02:49, 13.01s/it]
epoch [7/20], loss: 0.3130, test accuracy: 87.06%
training progress:  40%|████      | 8/20 [01:44<02:36, 13.05s/it]
epoch [8/20], loss: 0.3015, test accuracy: 87.35%
training progress:  45%|████▌     | 9/20 [01:57<02:24, 13.11s/it]
epoch [9/20], loss: 0.2931, test accuracy: 87.64%
training progress:  50%|█████     | 10/20 [02:10<02:11, 13.18s/it]
epoch [10/20], loss: 0.2843, test accuracy: 87.79%
training progress:  55%|█████▌    | 11/20 [02:23<01:58, 13.20s/it]
epoch [11/20], loss: 0.2810, test accuracy: 87.99%
training progress:  60%|██████    | 12/20 [02:37<01:45, 13.23s/it]
epoch [12/20], loss: 0.2748, test accuracy: 88.27%
training progress:  65%|██████▌   | 13/20 [02:50<01:32, 13.25s/it]
epoch [13/20], loss: 0.2678, test accuracy: 87.97%
training progress:  70%|███████   | 14/20 [03:03<01:19, 13.24s/it]
epoch [14/20], loss: 0.2644, test accuracy: 88.25%
training progress:  75%|███████▌  | 15/20 [03:17<01:06, 13.26s/it]
epoch [15/20], loss: 0.2558, test accuracy: 87.81%
training progress:  80%|████████  | 16/20 [03:30<00:53, 13.29s/it]
epoch [16/20], loss: 0.2530, test accuracy: 88.54%
training progress:  85%|████████▌ | 17/20 [03:43<00:40, 13.37s/it]
epoch [17/20], loss: 0.2474, test accuracy: 88.42%
training progress:  90%|█████████ | 18/20 [03:57<00:26, 13.35s/it]
epoch [18/20], loss: 0.2450, test accuracy: 88.42%
training progress:  95%|█████████▌| 19/20 [04:10<00:13, 13.32s/it]
epoch [19/20], loss: 0.2396, test accuracy: 89.17%
training progress: 100%|██████████| 20/20 [04:23<00:00, 13.18s/it]
epoch [20/20], loss: 0.2347, test accuracy: 88.76%

Training MLP model #3 with hidden layer design: [512, 256, 128]
training progress:   5%|▌         | 1/20 [00:13<04:15, 13.43s/it]
epoch [1/20], loss: 0.5530, test accuracy: 84.26%
training progress:  10%|█         | 2/20 [00:27<04:04, 13.58s/it]
epoch [2/20], loss: 0.4185, test accuracy: 85.00%
training progress:  15%|█▌        | 3/20 [00:40<03:50, 13.54s/it]
epoch [3/20], loss: 0.3809, test accuracy: 85.59%
training progress:  20%|██        | 4/20 [00:54<03:36, 13.51s/it]
epoch [4/20], loss: 0.3580, test accuracy: 85.83%
training progress:  25%|██▌       | 5/20 [01:07<03:22, 13.50s/it]
epoch [5/20], loss: 0.3387, test accuracy: 87.19%
training progress:  30%|███       | 6/20 [01:21<03:09, 13.56s/it]
epoch [6/20], loss: 0.3260, test accuracy: 86.30%
training progress:  35%|███▌      | 7/20 [01:34<02:56, 13.57s/it]
epoch [7/20], loss: 0.3199, test accuracy: 87.05%
training progress:  40%|████      | 8/20 [01:48<02:42, 13.56s/it]
epoch [8/20], loss: 0.3069, test accuracy: 87.56%
training progress:  45%|████▌     | 9/20 [02:01<02:29, 13.55s/it]
epoch [9/20], loss: 0.2995, test accuracy: 88.00%
training progress:  50%|█████     | 10/20 [02:15<02:14, 13.48s/it]
epoch [10/20], loss: 0.2915, test accuracy: 87.90%
training progress:  55%|█████▌    | 11/20 [02:28<02:01, 13.49s/it]
epoch [11/20], loss: 0.2853, test accuracy: 87.88%
training progress:  60%|██████    | 12/20 [02:42<01:47, 13.43s/it]
epoch [12/20], loss: 0.2749, test accuracy: 88.75%
training progress:  65%|██████▌   | 13/20 [02:55<01:34, 13.44s/it]
epoch [13/20], loss: 0.2733, test accuracy: 88.51%
training progress:  70%|███████   | 14/20 [03:08<01:20, 13.43s/it]
epoch [14/20], loss: 0.2652, test accuracy: 88.06%
training progress:  75%|███████▌  | 15/20 [03:22<01:07, 13.43s/it]
epoch [15/20], loss: 0.2606, test accuracy: 88.08%
training progress:  80%|████████  | 16/20 [03:35<00:53, 13.47s/it]
epoch [16/20], loss: 0.2559, test accuracy: 88.11%
training progress:  85%|████████▌ | 17/20 [03:49<00:40, 13.46s/it]
epoch [17/20], loss: 0.2479, test accuracy: 88.30%
training progress:  90%|█████████ | 18/20 [04:02<00:26, 13.46s/it]
epoch [18/20], loss: 0.2470, test accuracy: 88.49%
training progress:  95%|█████████▌| 19/20 [04:16<00:13, 13.44s/it]
epoch [19/20], loss: 0.2411, test accuracy: 88.86%
training progress: 100%|██████████| 20/20 [04:29<00:00, 13.49s/it]
epoch [20/20], loss: 0.2396, test accuracy: 88.59%

Training MLP model #4 with hidden layer design: [1024, 512, 256, 128]
training progress:   5%|▌         | 1/20 [00:13<04:18, 13.59s/it]
epoch [1/20], loss: 0.5786, test accuracy: 82.89%
training progress:  10%|█         | 2/20 [00:27<04:04, 13.59s/it]
epoch [2/20], loss: 0.4325, test accuracy: 85.28%
training progress:  15%|█▌        | 3/20 [00:40<03:52, 13.68s/it]
epoch [3/20], loss: 0.4001, test accuracy: 85.10%
training progress:  20%|██        | 4/20 [00:54<03:39, 13.70s/it]
epoch [4/20], loss: 0.3734, test accuracy: 86.40%
training progress:  25%|██▌       | 5/20 [01:08<03:25, 13.73s/it]
epoch [5/20], loss: 0.3492, test accuracy: 87.11%
training progress:  30%|███       | 6/20 [01:22<03:12, 13.74s/it]
epoch [6/20], loss: 0.3400, test accuracy: 87.76%
training progress:  35%|███▌      | 7/20 [01:35<02:58, 13.72s/it]
epoch [7/20], loss: 0.3235, test accuracy: 87.36%
training progress:  40%|████      | 8/20 [01:49<02:43, 13.61s/it]
epoch [8/20], loss: 0.3162, test accuracy: 87.73%
training progress:  45%|████▌     | 9/20 [02:02<02:30, 13.64s/it]
epoch [9/20], loss: 0.3090, test accuracy: 87.53%
training progress:  50%|█████     | 10/20 [02:16<02:16, 13.69s/it]
epoch [10/20], loss: 0.3004, test accuracy: 87.87%
training progress:  55%|█████▌    | 11/20 [02:30<02:03, 13.68s/it]
epoch [11/20], loss: 0.2934, test accuracy: 88.11%
training progress:  60%|██████    | 12/20 [02:44<01:49, 13.67s/it]
epoch [12/20], loss: 0.2845, test accuracy: 88.15%
training progress:  65%|██████▌   | 13/20 [02:57<01:35, 13.64s/it]
epoch [13/20], loss: 0.2800, test accuracy: 87.81%
training progress:  70%|███████   | 14/20 [03:11<01:21, 13.65s/it]
epoch [14/20], loss: 0.2750, test accuracy: 88.15%
training progress:  75%|███████▌  | 15/20 [03:24<01:07, 13.57s/it]
epoch [15/20], loss: 0.2644, test accuracy: 88.25%
training progress:  80%|████████  | 16/20 [03:38<00:54, 13.59s/it]
epoch [16/20], loss: 0.2619, test accuracy: 88.45%
training progress:  85%|████████▌ | 17/20 [03:52<00:41, 13.77s/it]
epoch [17/20], loss: 0.2597, test accuracy: 88.06%
training progress:  90%|█████████ | 18/20 [04:06<00:27, 13.74s/it]
epoch [18/20], loss: 0.2539, test accuracy: 88.59%
training progress:  95%|█████████▌| 19/20 [04:20<00:13, 13.76s/it]
epoch [19/20], loss: 0.2516, test accuracy: 88.72%
training progress: 100%|██████████| 20/20 [04:33<00:00, 13.67s/it]
epoch [20/20], loss: 0.2486, test accuracy: 88.28%

Training MLP model #5 with hidden layer design: [512, 1024, 256, 128]
training progress:   5%|▌         | 1/20 [00:13<04:20, 13.70s/it]
epoch [1/20], loss: 0.5760, test accuracy: 83.66%
training progress:  10%|█         | 2/20 [00:27<04:07, 13.75s/it]
epoch [2/20], loss: 0.4352, test accuracy: 84.52%
training progress:  15%|█▌        | 3/20 [00:41<03:52, 13.69s/it]
epoch [3/20], loss: 0.3986, test accuracy: 85.22%
training progress:  20%|██        | 4/20 [00:54<03:38, 13.69s/it]
epoch [4/20], loss: 0.3750, test accuracy: 85.76%
training progress:  25%|██▌       | 5/20 [01:08<03:26, 13.73s/it]
epoch [5/20], loss: 0.3530, test accuracy: 86.91%
training progress:  30%|███       | 6/20 [01:22<03:12, 13.76s/it]
epoch [6/20], loss: 0.3456, test accuracy: 86.87%
training progress:  35%|███▌      | 7/20 [01:35<02:57, 13.69s/it]
epoch [7/20], loss: 0.3291, test accuracy: 86.82%
training progress:  40%|████      | 8/20 [01:49<02:43, 13.64s/it]
epoch [8/20], loss: 0.3218, test accuracy: 87.59%
training progress:  45%|████▌     | 9/20 [02:02<02:28, 13.54s/it]
epoch [9/20], loss: 0.3107, test accuracy: 87.99%
training progress:  50%|█████     | 10/20 [02:16<02:15, 13.50s/it]
epoch [10/20], loss: 0.3052, test accuracy: 87.64%
training progress:  55%|█████▌    | 11/20 [02:29<02:01, 13.54s/it]
epoch [11/20], loss: 0.3002, test accuracy: 87.56%
training progress:  60%|██████    | 12/20 [02:43<01:48, 13.55s/it]
epoch [12/20], loss: 0.2921, test accuracy: 87.44%
training progress:  65%|██████▌   | 13/20 [02:57<01:35, 13.58s/it]
epoch [13/20], loss: 0.2849, test accuracy: 88.39%
training progress:  70%|███████   | 14/20 [03:10<01:21, 13.54s/it]
epoch [14/20], loss: 0.2807, test accuracy: 88.35%
training progress:  75%|███████▌  | 15/20 [03:24<01:07, 13.56s/it]
epoch [15/20], loss: 0.2729, test accuracy: 88.58%
training progress:  80%|████████  | 16/20 [03:37<00:54, 13.54s/it]
epoch [16/20], loss: 0.2670, test accuracy: 88.15%
training progress:  85%|████████▌ | 17/20 [03:51<00:40, 13.53s/it]
epoch [17/20], loss: 0.2652, test accuracy: 87.90%
training progress:  90%|█████████ | 18/20 [04:04<00:27, 13.53s/it]
epoch [18/20], loss: 0.2629, test accuracy: 88.06%
training progress:  95%|█████████▌| 19/20 [04:18<00:13, 13.57s/it]
epoch [19/20], loss: 0.2610, test accuracy: 88.91%
training progress: 100%|██████████| 20/20 [04:31<00:00, 13.60s/it]
epoch [20/20], loss: 0.2529, test accuracy: 88.41%
```
