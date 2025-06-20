use device: cuda

============================================================
test config: AlexNet_Original
============================================================
total params: 58,299,082
trainable params: 58,299,082
config AlexNet_Original failed: Given input size: (256x2x2). Calculated output size: (256x0x0). Output size is too small

============================================================
test config: ModifiedAlexNet_v1
============================================================
total params: 8,378,890
trainable params: 8,378,890
Epoch 1/30, Batch 0/469, Loss: 2.3026
Epoch 1/30, Batch 100/469, Loss: 0.5554
Epoch 1/30, Batch 200/469, Loss: 0.6357
Epoch 1/30, Batch 300/469, Loss: 0.6090
Epoch 1/30, Batch 400/469, Loss: 0.4049
Epoch [1/30], Loss: 0.6360, Train Acc: 76.11%, Test Acc: 84.09%, LR: 0.001000
Epoch 2/30, Batch 0/469, Loss: 0.4450
Epoch 2/30, Batch 100/469, Loss: 0.4636
Epoch 2/30, Batch 200/469, Loss: 0.2858
Epoch 2/30, Batch 300/469, Loss: 0.3838
Epoch 2/30, Batch 400/469, Loss: 0.4064
Epoch [2/30], Loss: 0.3900, Train Acc: 85.65%, Test Acc: 87.46%, LR: 0.001000
Epoch 3/30, Batch 0/469, Loss: 0.4253
Epoch 3/30, Batch 100/469, Loss: 0.5343
Epoch 3/30, Batch 200/469, Loss: 0.4079
Epoch 3/30, Batch 300/469, Loss: 0.3088
Epoch 3/30, Batch 400/469, Loss: 0.3603
Epoch [3/30], Loss: 0.3419, Train Acc: 87.74%, Test Acc: 89.26%, LR: 0.001000
Epoch 4/30, Batch 0/469, Loss: 0.2281
Epoch 4/30, Batch 100/469, Loss: 0.4586
Epoch 4/30, Batch 200/469, Loss: 0.2792
Epoch 4/30, Batch 300/469, Loss: 0.4000
Epoch 4/30, Batch 400/469, Loss: 0.3790
Epoch [4/30], Loss: 0.3096, Train Acc: 88.77%, Test Acc: 89.60%, LR: 0.001000
Epoch 5/30, Batch 0/469, Loss: 0.3336
Epoch 5/30, Batch 100/469, Loss: 0.3016
Epoch 5/30, Batch 200/469, Loss: 0.4297
Epoch 5/30, Batch 300/469, Loss: 0.2924
Epoch 5/30, Batch 400/469, Loss: 0.2853
Epoch [5/30], Loss: 0.2918, Train Acc: 89.52%, Test Acc: 89.96%, LR: 0.001000
Epoch 6/30, Batch 0/469, Loss: 0.3223
Epoch 6/30, Batch 100/469, Loss: 0.2863
Epoch 6/30, Batch 200/469, Loss: 0.2031
Epoch 6/30, Batch 300/469, Loss: 0.3096
Epoch 6/30, Batch 400/469, Loss: 0.2634
Epoch [6/30], Loss: 0.2790, Train Acc: 90.13%, Test Acc: 90.63%, LR: 0.001000
Epoch 7/30, Batch 0/469, Loss: 0.2129
Epoch 7/30, Batch 100/469, Loss: 0.2818
Epoch 7/30, Batch 200/469, Loss: 0.2988
Epoch 7/30, Batch 300/469, Loss: 0.2284
Epoch 7/30, Batch 400/469, Loss: 0.1872
Epoch [7/30], Loss: 0.2656, Train Acc: 90.58%, Test Acc: 91.09%, LR: 0.001000
Epoch 8/30, Batch 0/469, Loss: 0.2260
Epoch 8/30, Batch 100/469, Loss: 0.1691
Epoch 8/30, Batch 200/469, Loss: 0.1749
Epoch 8/30, Batch 300/469, Loss: 0.2162
Epoch 8/30, Batch 400/469, Loss: 0.2171
Epoch [8/30], Loss: 0.2166, Train Acc: 92.06%, Test Acc: 92.06%, LR: 0.000100
Epoch 9/30, Batch 0/469, Loss: 0.1636
Epoch 9/30, Batch 100/469, Loss: 0.1816
Epoch 9/30, Batch 200/469, Loss: 0.2114
Epoch 9/30, Batch 300/469, Loss: 0.1502
Epoch 9/30, Batch 400/469, Loss: 0.1336
Epoch [9/30], Loss: 0.2021, Train Acc: 92.67%, Test Acc: 92.37%, LR: 0.000100
Epoch 10/30, Batch 0/469, Loss: 0.2196
Epoch 10/30, Batch 100/469, Loss: 0.1273
Epoch 10/30, Batch 200/469, Loss: 0.0881
Epoch 10/30, Batch 300/469, Loss: 0.2230
Epoch 10/30, Batch 400/469, Loss: 0.1532
Epoch [10/30], Loss: 0.1960, Train Acc: 92.80%, Test Acc: 92.40%, LR: 0.000100
Epoch 11/30, Batch 0/469, Loss: 0.2420
Epoch 11/30, Batch 100/469, Loss: 0.1270
Epoch 11/30, Batch 200/469, Loss: 0.1645
Epoch 11/30, Batch 300/469, Loss: 0.2063
Epoch 11/30, Batch 400/469, Loss: 0.1528
Epoch [11/30], Loss: 0.1887, Train Acc: 93.05%, Test Acc: 92.33%, LR: 0.000100
Epoch 12/30, Batch 0/469, Loss: 0.1292
Epoch 12/30, Batch 100/469, Loss: 0.0880
Epoch 12/30, Batch 200/469, Loss: 0.1747
Epoch 12/30, Batch 300/469, Loss: 0.2846
Epoch 12/30, Batch 400/469, Loss: 0.1055
Epoch [12/30], Loss: 0.1848, Train Acc: 93.22%, Test Acc: 92.33%, LR: 0.000100
Epoch 13/30, Batch 0/469, Loss: 0.0952
Epoch 13/30, Batch 100/469, Loss: 0.1501
Epoch 13/30, Batch 200/469, Loss: 0.1674
Epoch 13/30, Batch 300/469, Loss: 0.1915
Epoch 13/30, Batch 400/469, Loss: 0.1626
Epoch [13/30], Loss: 0.1798, Train Acc: 93.39%, Test Acc: 92.74%, LR: 0.000100
Epoch 14/30, Batch 0/469, Loss: 0.1555
Epoch 14/30, Batch 100/469, Loss: 0.1356
Epoch 14/30, Batch 200/469, Loss: 0.1832
Epoch 14/30, Batch 300/469, Loss: 0.1504
Epoch 14/30, Batch 400/469, Loss: 0.1464
Epoch [14/30], Loss: 0.1756, Train Acc: 93.54%, Test Acc: 92.64%, LR: 0.000100
Epoch 15/30, Batch 0/469, Loss: 0.2465
Epoch 15/30, Batch 100/469, Loss: 0.2242
Epoch 15/30, Batch 200/469, Loss: 0.1078
Epoch 15/30, Batch 300/469, Loss: 0.1796
Epoch 15/30, Batch 400/469, Loss: 0.2241
Epoch [15/30], Loss: 0.1671, Train Acc: 93.84%, Test Acc: 92.75%, LR: 0.000010
Epoch 16/30, Batch 0/469, Loss: 0.1242
Epoch 16/30, Batch 100/469, Loss: 0.2232
Epoch 16/30, Batch 200/469, Loss: 0.2544
Epoch 16/30, Batch 300/469, Loss: 0.0692
Epoch 16/30, Batch 400/469, Loss: 0.2365
Epoch [16/30], Loss: 0.1648, Train Acc: 93.91%, Test Acc: 92.72%, LR: 0.000010
Epoch 17/30, Batch 0/469, Loss: 0.2883
Epoch 17/30, Batch 100/469, Loss: 0.1410
Epoch 17/30, Batch 200/469, Loss: 0.1347
Epoch 17/30, Batch 300/469, Loss: 0.2009
Epoch 17/30, Batch 400/469, Loss: 0.1264
Epoch [17/30], Loss: 0.1641, Train Acc: 93.89%, Test Acc: 92.77%, LR: 0.000010
Epoch 18/30, Batch 0/469, Loss: 0.2609
Epoch 18/30, Batch 100/469, Loss: 0.1056
Epoch 18/30, Batch 200/469, Loss: 0.0891
Epoch 18/30, Batch 300/469, Loss: 0.1817
Epoch 18/30, Batch 400/469, Loss: 0.1685
Epoch [18/30], Loss: 0.1655, Train Acc: 93.98%, Test Acc: 92.81%, LR: 0.000010
Epoch 19/30, Batch 0/469, Loss: 0.1668
Epoch 19/30, Batch 100/469, Loss: 0.2631
Epoch 19/30, Batch 200/469, Loss: 0.1960
Epoch 19/30, Batch 300/469, Loss: 0.1483
Epoch 19/30, Batch 400/469, Loss: 0.1477
Epoch [19/30], Loss: 0.1633, Train Acc: 93.98%, Test Acc: 92.78%, LR: 0.000010
Epoch 20/30, Batch 0/469, Loss: 0.1287
Epoch 20/30, Batch 100/469, Loss: 0.1206
Epoch 20/30, Batch 200/469, Loss: 0.1463
Epoch 20/30, Batch 300/469, Loss: 0.1195
Epoch 20/30, Batch 400/469, Loss: 0.1302
Epoch [20/30], Loss: 0.1624, Train Acc: 94.00%, Test Acc: 92.85%, LR: 0.000010
Epoch 21/30, Batch 0/469, Loss: 0.1554
Epoch 21/30, Batch 100/469, Loss: 0.0648
Epoch 21/30, Batch 200/469, Loss: 0.1651
Epoch 21/30, Batch 300/469, Loss: 0.0998
Epoch 21/30, Batch 400/469, Loss: 0.1256
Epoch [21/30], Loss: 0.1620, Train Acc: 94.06%, Test Acc: 92.81%, LR: 0.000010
Epoch 22/30, Batch 0/469, Loss: 0.1313
Epoch 22/30, Batch 100/469, Loss: 0.1991
Epoch 22/30, Batch 200/469, Loss: 0.1243
Epoch 22/30, Batch 300/469, Loss: 0.1299
Epoch 22/30, Batch 400/469, Loss: 0.1773
Epoch [22/30], Loss: 0.1604, Train Acc: 94.03%, Test Acc: 92.86%, LR: 0.000001
Epoch 23/30, Batch 0/469, Loss: 0.1837
Epoch 23/30, Batch 100/469, Loss: 0.1996
Epoch 23/30, Batch 200/469, Loss: 0.1561
Epoch 23/30, Batch 300/469, Loss: 0.1012
Epoch 23/30, Batch 400/469, Loss: 0.1941
Epoch [23/30], Loss: 0.1593, Train Acc: 94.00%, Test Acc: 92.87%, LR: 0.000001
Epoch 24/30, Batch 0/469, Loss: 0.1561
Epoch 24/30, Batch 100/469, Loss: 0.1182
Epoch 24/30, Batch 200/469, Loss: 0.1448
Epoch 24/30, Batch 300/469, Loss: 0.1194
Epoch 24/30, Batch 400/469, Loss: 0.1382
Epoch [24/30], Loss: 0.1607, Train Acc: 94.08%, Test Acc: 92.87%, LR: 0.000001
Epoch 25/30, Batch 0/469, Loss: 0.1619
Epoch 25/30, Batch 100/469, Loss: 0.1494
Epoch 25/30, Batch 200/469, Loss: 0.0986
Epoch 25/30, Batch 300/469, Loss: 0.1911
Epoch 25/30, Batch 400/469, Loss: 0.1147
Epoch [25/30], Loss: 0.1594, Train Acc: 94.08%, Test Acc: 92.87%, LR: 0.000001
Epoch 26/30, Batch 0/469, Loss: 0.1959
Epoch 26/30, Batch 100/469, Loss: 0.1927
Epoch 26/30, Batch 200/469, Loss: 0.1184
Epoch 26/30, Batch 300/469, Loss: 0.0876
Epoch 26/30, Batch 400/469, Loss: 0.1212
Epoch [26/30], Loss: 0.1606, Train Acc: 94.03%, Test Acc: 92.85%, LR: 0.000001
Epoch 27/30, Batch 0/469, Loss: 0.0914
Epoch 27/30, Batch 100/469, Loss: 0.0894
Epoch 27/30, Batch 200/469, Loss: 0.1773
Epoch 27/30, Batch 300/469, Loss: 0.1390
Epoch 27/30, Batch 400/469, Loss: 0.0995
Epoch [27/30], Loss: 0.1612, Train Acc: 93.98%, Test Acc: 92.84%, LR: 0.000001
Epoch 28/30, Batch 0/469, Loss: 0.1812
Epoch 28/30, Batch 100/469, Loss: 0.1305
Epoch 28/30, Batch 200/469, Loss: 0.2140
Epoch 28/30, Batch 300/469, Loss: 0.1521
Epoch 28/30, Batch 400/469, Loss: 0.1359
Epoch [28/30], Loss: 0.1588, Train Acc: 94.13%, Test Acc: 92.85%, LR: 0.000001
Epoch 29/30, Batch 0/469, Loss: 0.1111
Epoch 29/30, Batch 100/469, Loss: 0.1427
Epoch 29/30, Batch 200/469, Loss: 0.1711
Epoch 29/30, Batch 300/469, Loss: 0.2171
Epoch 29/30, Batch 400/469, Loss: 0.0863
Epoch [29/30], Loss: 0.1596, Train Acc: 94.18%, Test Acc: 92.83%, LR: 0.000000
Epoch 30/30, Batch 0/469, Loss: 0.1259
Epoch 30/30, Batch 100/469, Loss: 0.1518
Epoch 30/30, Batch 200/469, Loss: 0.2312
Epoch 30/30, Batch 300/469, Loss: 0.1764
Epoch 30/30, Batch 400/469, Loss: 0.1414
Epoch [30/30], Loss: 0.1604, Train Acc: 94.09%, Test Acc: 92.84%, LR: 0.000000
load best model, test accuracy: 92.87%
training time: 193.20s
best test accuracy: 92.87%
final test accuracy: 92.84%

============================================================
test config: ModifiedAlexNet_v2
============================================================
total params: 8,378,890
trainable params: 8,378,890
Epoch 1/30, Batch 0/938, Loss: 2.3019
Epoch 1/30, Batch 100/938, Loss: 0.7573
Epoch 1/30, Batch 200/938, Loss: 0.4634
Epoch 1/30, Batch 300/938, Loss: 0.6962
Epoch 1/30, Batch 400/938, Loss: 0.4101
Epoch 1/30, Batch 500/938, Loss: 0.3886
Epoch 1/30, Batch 600/938, Loss: 0.5668
Epoch 1/30, Batch 700/938, Loss: 0.3894
Epoch 1/30, Batch 800/938, Loss: 0.4079
Epoch 1/30, Batch 900/938, Loss: 0.3424
Epoch [1/30], Loss: 0.5740, Train Acc: 78.34%, Test Acc: 82.43%, LR: 0.000500
Epoch 2/30, Batch 0/938, Loss: 0.3466
Epoch 2/30, Batch 100/938, Loss: 0.4427
Epoch 2/30, Batch 200/938, Loss: 0.2160
Epoch 2/30, Batch 300/938, Loss: 0.3723
Epoch 2/30, Batch 400/938, Loss: 0.6146
Epoch 2/30, Batch 500/938, Loss: 0.2851
Epoch 2/30, Batch 600/938, Loss: 0.4106
Epoch 2/30, Batch 700/938, Loss: 0.3419
Epoch 2/30, Batch 800/938, Loss: 0.3024
Epoch 2/30, Batch 900/938, Loss: 0.5079
Epoch [2/30], Loss: 0.3736, Train Acc: 86.29%, Test Acc: 86.73%, LR: 0.000500
Epoch 3/30, Batch 0/938, Loss: 0.2848
Epoch 3/30, Batch 100/938, Loss: 0.2279
Epoch 3/30, Batch 200/938, Loss: 0.2317
Epoch 3/30, Batch 300/938, Loss: 0.2903
Epoch 3/30, Batch 400/938, Loss: 0.4026
Epoch 3/30, Batch 500/938, Loss: 0.3733
Epoch 3/30, Batch 600/938, Loss: 0.3335
Epoch 3/30, Batch 700/938, Loss: 0.2853
Epoch 3/30, Batch 800/938, Loss: 0.4831
Epoch 3/30, Batch 900/938, Loss: 0.1292
Epoch [3/30], Loss: 0.3317, Train Acc: 87.76%, Test Acc: 88.72%, LR: 0.000500
Epoch 4/30, Batch 0/938, Loss: 0.2901
Epoch 4/30, Batch 100/938, Loss: 0.3498
Epoch 4/30, Batch 200/938, Loss: 0.2731
Epoch 4/30, Batch 300/938, Loss: 0.2705
Epoch 4/30, Batch 400/938, Loss: 0.2715
Epoch 4/30, Batch 500/938, Loss: 0.4153
Epoch 4/30, Batch 600/938, Loss: 0.3085
Epoch 4/30, Batch 700/938, Loss: 0.2724
Epoch 4/30, Batch 800/938, Loss: 0.2846
Epoch 4/30, Batch 900/938, Loss: 0.4867
Epoch [4/30], Loss: 0.3040, Train Acc: 88.85%, Test Acc: 89.91%, LR: 0.000500
Epoch 5/30, Batch 0/938, Loss: 0.2183
Epoch 5/30, Batch 100/938, Loss: 0.2096
Epoch 5/30, Batch 200/938, Loss: 0.2349
Epoch 5/30, Batch 300/938, Loss: 0.4380
Epoch 5/30, Batch 400/938, Loss: 0.2577
Epoch 5/30, Batch 500/938, Loss: 0.2797
Epoch 5/30, Batch 600/938, Loss: 0.2453
Epoch 5/30, Batch 700/938, Loss: 0.2950
Epoch 5/30, Batch 800/938, Loss: 0.1765
Epoch 5/30, Batch 900/938, Loss: 0.3182
Epoch [5/30], Loss: 0.2904, Train Acc: 89.42%, Test Acc: 90.02%, LR: 0.000500
Epoch 6/30, Batch 0/938, Loss: 0.2379
Epoch 6/30, Batch 100/938, Loss: 0.4024
Epoch 6/30, Batch 200/938, Loss: 0.4980
Epoch 6/30, Batch 300/938, Loss: 0.2269
Epoch 6/30, Batch 400/938, Loss: 0.2151
Epoch 6/30, Batch 500/938, Loss: 0.2250
Epoch 6/30, Batch 600/938, Loss: 0.4749
Epoch 6/30, Batch 700/938, Loss: 0.3799
Epoch 6/30, Batch 800/938, Loss: 0.2589
Epoch 6/30, Batch 900/938, Loss: 0.1490
Epoch [6/30], Loss: 0.2796, Train Acc: 89.86%, Test Acc: 90.21%, LR: 0.000500
Epoch 7/30, Batch 0/938, Loss: 0.2040
Epoch 7/30, Batch 100/938, Loss: 0.2872
Epoch 7/30, Batch 200/938, Loss: 0.1887
Epoch 7/30, Batch 300/938, Loss: 0.2521
Epoch 7/30, Batch 400/938, Loss: 0.3800
Epoch 7/30, Batch 500/938, Loss: 0.1870
Epoch 7/30, Batch 600/938, Loss: 0.3890
Epoch 7/30, Batch 700/938, Loss: 0.2474
Epoch 7/30, Batch 800/938, Loss: 0.1504
Epoch 7/30, Batch 900/938, Loss: 0.2814
Epoch [7/30], Loss: 0.2701, Train Acc: 90.12%, Test Acc: 90.31%, LR: 0.000500
Epoch 8/30, Batch 0/938, Loss: 0.2161
Epoch 8/30, Batch 100/938, Loss: 0.1677
Epoch 8/30, Batch 200/938, Loss: 0.1948
Epoch 8/30, Batch 300/938, Loss: 0.1366
Epoch 8/30, Batch 400/938, Loss: 0.1261
Epoch 8/30, Batch 500/938, Loss: 0.2290
Epoch 8/30, Batch 600/938, Loss: 0.1705
Epoch 8/30, Batch 700/938, Loss: 0.2558
Epoch 8/30, Batch 800/938, Loss: 0.1083
Epoch 8/30, Batch 900/938, Loss: 0.1245
Epoch [8/30], Loss: 0.2185, Train Acc: 92.00%, Test Acc: 92.28%, LR: 0.000050
Epoch 9/30, Batch 0/938, Loss: 0.1853
Epoch 9/30, Batch 100/938, Loss: 0.2411
Epoch 9/30, Batch 200/938, Loss: 0.3437
Epoch 9/30, Batch 300/938, Loss: 0.1455
Epoch 9/30, Batch 400/938, Loss: 0.1527
Epoch 9/30, Batch 500/938, Loss: 0.3783
Epoch 9/30, Batch 600/938, Loss: 0.1732
Epoch 9/30, Batch 700/938, Loss: 0.1640
Epoch 9/30, Batch 800/938, Loss: 0.1833
Epoch 9/30, Batch 900/938, Loss: 0.2077
Epoch [9/30], Loss: 0.2077, Train Acc: 92.31%, Test Acc: 92.30%, LR: 0.000050
Epoch 10/30, Batch 0/938, Loss: 0.2049
Epoch 10/30, Batch 100/938, Loss: 0.2212
Epoch 10/30, Batch 200/938, Loss: 0.1499
Epoch 10/30, Batch 300/938, Loss: 0.1282
Epoch 10/30, Batch 400/938, Loss: 0.1033
Epoch 10/30, Batch 500/938, Loss: 0.2233
Epoch 10/30, Batch 600/938, Loss: 0.0910
Epoch 10/30, Batch 700/938, Loss: 0.2572
Epoch 10/30, Batch 800/938, Loss: 0.2610
Epoch 10/30, Batch 900/938, Loss: 0.2506
Epoch [10/30], Loss: 0.2045, Train Acc: 92.51%, Test Acc: 92.57%, LR: 0.000050
Epoch 11/30, Batch 0/938, Loss: 0.0875
Epoch 11/30, Batch 100/938, Loss: 0.2228
Epoch 11/30, Batch 200/938, Loss: 0.0770
Epoch 11/30, Batch 300/938, Loss: 0.1254
Epoch 11/30, Batch 400/938, Loss: 0.0635
Epoch 11/30, Batch 500/938, Loss: 0.2796
Epoch 11/30, Batch 600/938, Loss: 0.2237
Epoch 11/30, Batch 700/938, Loss: 0.1229
Epoch 11/30, Batch 800/938, Loss: 0.2118
Epoch 11/30, Batch 900/938, Loss: 0.1102
Epoch [11/30], Loss: 0.2011, Train Acc: 92.64%, Test Acc: 92.52%, LR: 0.000050
Epoch 12/30, Batch 0/938, Loss: 0.1964
Epoch 12/30, Batch 100/938, Loss: 0.1570
Epoch 12/30, Batch 200/938, Loss: 0.1929
Epoch 12/30, Batch 300/938, Loss: 0.2201
Epoch 12/30, Batch 400/938, Loss: 0.1724
Epoch 12/30, Batch 500/938, Loss: 0.0954
Epoch 12/30, Batch 600/938, Loss: 0.1683
Epoch 12/30, Batch 700/938, Loss: 0.2238
Epoch 12/30, Batch 800/938, Loss: 0.2058
Epoch 12/30, Batch 900/938, Loss: 0.0750
Epoch [12/30], Loss: 0.1970, Train Acc: 92.72%, Test Acc: 92.67%, LR: 0.000050
Epoch 13/30, Batch 0/938, Loss: 0.2908
Epoch 13/30, Batch 100/938, Loss: 0.2317
Epoch 13/30, Batch 200/938, Loss: 0.2018
Epoch 13/30, Batch 300/938, Loss: 0.3445
Epoch 13/30, Batch 400/938, Loss: 0.2106
Epoch 13/30, Batch 500/938, Loss: 0.1150
Epoch 13/30, Batch 600/938, Loss: 0.2440
Epoch 13/30, Batch 700/938, Loss: 0.2270
Epoch 13/30, Batch 800/938, Loss: 0.3404
Epoch 13/30, Batch 900/938, Loss: 0.0959
Epoch [13/30], Loss: 0.1941, Train Acc: 92.96%, Test Acc: 92.34%, LR: 0.000050
Epoch 14/30, Batch 0/938, Loss: 0.1060
Epoch 14/30, Batch 100/938, Loss: 0.0468
Epoch 14/30, Batch 200/938, Loss: 0.1934
Epoch 14/30, Batch 300/938, Loss: 0.1621
Epoch 14/30, Batch 400/938, Loss: 0.0983
Epoch 14/30, Batch 500/938, Loss: 0.1426
Epoch 14/30, Batch 600/938, Loss: 0.3047
Epoch 14/30, Batch 700/938, Loss: 0.2924
Epoch 14/30, Batch 800/938, Loss: 0.1437
Epoch 14/30, Batch 900/938, Loss: 0.2110
Epoch [14/30], Loss: 0.1939, Train Acc: 92.98%, Test Acc: 92.85%, LR: 0.000050
Epoch 15/30, Batch 0/938, Loss: 0.1089
Epoch 15/30, Batch 100/938, Loss: 0.1424
Epoch 15/30, Batch 200/938, Loss: 0.1840
Epoch 15/30, Batch 300/938, Loss: 0.1851
Epoch 15/30, Batch 400/938, Loss: 0.1475
Epoch 15/30, Batch 500/938, Loss: 0.1014
Epoch 15/30, Batch 600/938, Loss: 0.1761
Epoch 15/30, Batch 700/938, Loss: 0.1137
Epoch 15/30, Batch 800/938, Loss: 0.0989
Epoch 15/30, Batch 900/938, Loss: 0.1416
Epoch [15/30], Loss: 0.1829, Train Acc: 93.38%, Test Acc: 92.85%, LR: 0.000005
Epoch 16/30, Batch 0/938, Loss: 0.1188
Epoch 16/30, Batch 100/938, Loss: 0.3136
Epoch 16/30, Batch 200/938, Loss: 0.2413
Epoch 16/30, Batch 300/938, Loss: 0.2409
Epoch 16/30, Batch 400/938, Loss: 0.0985
Epoch 16/30, Batch 500/938, Loss: 0.1340
Epoch 16/30, Batch 600/938, Loss: 0.1534
Epoch 16/30, Batch 700/938, Loss: 0.1827
Epoch 16/30, Batch 800/938, Loss: 0.1301
Epoch 16/30, Batch 900/938, Loss: 0.0946
Epoch [16/30], Loss: 0.1841, Train Acc: 93.31%, Test Acc: 93.00%, LR: 0.000005
Epoch 17/30, Batch 0/938, Loss: 0.1468
Epoch 17/30, Batch 100/938, Loss: 0.2893
Epoch 17/30, Batch 200/938, Loss: 0.1092
Epoch 17/30, Batch 300/938, Loss: 0.1294
Epoch 17/30, Batch 400/938, Loss: 0.3779
Epoch 17/30, Batch 500/938, Loss: 0.2243
Epoch 17/30, Batch 600/938, Loss: 0.2494
Epoch 17/30, Batch 700/938, Loss: 0.1127
Epoch 17/30, Batch 800/938, Loss: 0.1643
Epoch 17/30, Batch 900/938, Loss: 0.1360
Epoch [17/30], Loss: 0.1819, Train Acc: 93.40%, Test Acc: 92.85%, LR: 0.000005
Epoch 18/30, Batch 0/938, Loss: 0.2469
Epoch 18/30, Batch 100/938, Loss: 0.1289
Epoch 18/30, Batch 200/938, Loss: 0.2690
Epoch 18/30, Batch 300/938, Loss: 0.1750
Epoch 18/30, Batch 400/938, Loss: 0.4425
Epoch 18/30, Batch 500/938, Loss: 0.2205
Epoch 18/30, Batch 600/938, Loss: 0.3501
Epoch 18/30, Batch 700/938, Loss: 0.2065
Epoch 18/30, Batch 800/938, Loss: 0.1131
Epoch 18/30, Batch 900/938, Loss: 0.1326
Epoch [18/30], Loss: 0.1810, Train Acc: 93.43%, Test Acc: 92.94%, LR: 0.000005
Epoch 19/30, Batch 0/938, Loss: 0.1714
Epoch 19/30, Batch 100/938, Loss: 0.2291
Epoch 19/30, Batch 200/938, Loss: 0.2292
Epoch 19/30, Batch 300/938, Loss: 0.1636
Epoch 19/30, Batch 400/938, Loss: 0.1331
Epoch 19/30, Batch 500/938, Loss: 0.1079
Epoch 19/30, Batch 600/938, Loss: 0.2069
Epoch 19/30, Batch 700/938, Loss: 0.1659
Epoch 19/30, Batch 800/938, Loss: 0.1464
Epoch 19/30, Batch 900/938, Loss: 0.3868
Epoch [19/30], Loss: 0.1818, Train Acc: 93.39%, Test Acc: 93.00%, LR: 0.000005
Epoch 20/30, Batch 0/938, Loss: 0.2502
Epoch 20/30, Batch 100/938, Loss: 0.2702
Epoch 20/30, Batch 200/938, Loss: 0.2348
Epoch 20/30, Batch 300/938, Loss: 0.1231
Epoch 20/30, Batch 400/938, Loss: 0.1715
Epoch 20/30, Batch 500/938, Loss: 0.2365
Epoch 20/30, Batch 600/938, Loss: 0.1998
Epoch 20/30, Batch 700/938, Loss: 0.1447
Epoch 20/30, Batch 800/938, Loss: 0.2815
Epoch 20/30, Batch 900/938, Loss: 0.0806
Epoch [20/30], Loss: 0.1801, Train Acc: 93.46%, Test Acc: 93.01%, LR: 0.000005
Epoch 21/30, Batch 0/938, Loss: 0.2932
Epoch 21/30, Batch 100/938, Loss: 0.1848
Epoch 21/30, Batch 200/938, Loss: 0.1602
Epoch 21/30, Batch 300/938, Loss: 0.1051
Epoch 21/30, Batch 400/938, Loss: 0.2150
Epoch 21/30, Batch 500/938, Loss: 0.1861
Epoch 21/30, Batch 600/938, Loss: 0.1535
Epoch 21/30, Batch 700/938, Loss: 0.1057
Epoch 21/30, Batch 800/938, Loss: 0.1880
Epoch 21/30, Batch 900/938, Loss: 0.1717
Epoch [21/30], Loss: 0.1790, Train Acc: 93.40%, Test Acc: 93.13%, LR: 0.000005
Epoch 22/30, Batch 0/938, Loss: 0.2099
Epoch 22/30, Batch 100/938, Loss: 0.1536
Epoch 22/30, Batch 200/938, Loss: 0.2409
Epoch 22/30, Batch 300/938, Loss: 0.0786
Epoch 22/30, Batch 400/938, Loss: 0.1059
Epoch 22/30, Batch 500/938, Loss: 0.1044
Epoch 22/30, Batch 600/938, Loss: 0.1229
Epoch 22/30, Batch 700/938, Loss: 0.2534
Epoch 22/30, Batch 800/938, Loss: 0.1067
Epoch 22/30, Batch 900/938, Loss: 0.0937
Epoch [22/30], Loss: 0.1802, Train Acc: 93.44%, Test Acc: 93.04%, LR: 0.000001
Epoch 23/30, Batch 0/938, Loss: 0.1508
Epoch 23/30, Batch 100/938, Loss: 0.2787
Epoch 23/30, Batch 200/938, Loss: 0.1519
Epoch 23/30, Batch 300/938, Loss: 0.1772
Epoch 23/30, Batch 400/938, Loss: 0.3015
Epoch 23/30, Batch 500/938, Loss: 0.1443
Epoch 23/30, Batch 600/938, Loss: 0.2377
Epoch 23/30, Batch 700/938, Loss: 0.1470
Epoch 23/30, Batch 800/938, Loss: 0.2211
Epoch 23/30, Batch 900/938, Loss: 0.0603
Epoch [23/30], Loss: 0.1796, Train Acc: 93.40%, Test Acc: 93.02%, LR: 0.000001
Epoch 24/30, Batch 0/938, Loss: 0.1852
Epoch 24/30, Batch 100/938, Loss: 0.1654
Epoch 24/30, Batch 200/938, Loss: 0.0715
Epoch 24/30, Batch 300/938, Loss: 0.1662
Epoch 24/30, Batch 400/938, Loss: 0.1302
Epoch 24/30, Batch 500/938, Loss: 0.1178
Epoch 24/30, Batch 600/938, Loss: 0.2073
Epoch 24/30, Batch 700/938, Loss: 0.1079
Epoch 24/30, Batch 800/938, Loss: 0.2082
Epoch 24/30, Batch 900/938, Loss: 0.2773
Epoch [24/30], Loss: 0.1780, Train Acc: 93.53%, Test Acc: 93.01%, LR: 0.000001
Epoch 25/30, Batch 0/938, Loss: 0.1181
Epoch 25/30, Batch 100/938, Loss: 0.1107
Epoch 25/30, Batch 200/938, Loss: 0.2727
Epoch 25/30, Batch 300/938, Loss: 0.1414
Epoch 25/30, Batch 400/938, Loss: 0.2804
Epoch 25/30, Batch 500/938, Loss: 0.1033
Epoch 25/30, Batch 600/938, Loss: 0.1879
Epoch 25/30, Batch 700/938, Loss: 0.2036
Epoch 25/30, Batch 800/938, Loss: 0.1559
Epoch 25/30, Batch 900/938, Loss: 0.2004
Epoch [25/30], Loss: 0.1786, Train Acc: 93.42%, Test Acc: 92.99%, LR: 0.000001
Epoch 26/30, Batch 0/938, Loss: 0.1415
Epoch 26/30, Batch 100/938, Loss: 0.2003
Epoch 26/30, Batch 200/938, Loss: 0.2354
Epoch 26/30, Batch 300/938, Loss: 0.1484
Epoch 26/30, Batch 400/938, Loss: 0.2221
Epoch 26/30, Batch 500/938, Loss: 0.3331
Epoch 26/30, Batch 600/938, Loss: 0.1974
Epoch 26/30, Batch 700/938, Loss: 0.0746
Epoch 26/30, Batch 800/938, Loss: 0.2011
Epoch 26/30, Batch 900/938, Loss: 0.1531
Epoch [26/30], Loss: 0.1792, Train Acc: 93.51%, Test Acc: 92.98%, LR: 0.000001
Epoch 27/30, Batch 0/938, Loss: 0.1570
Epoch 27/30, Batch 100/938, Loss: 0.3527
Epoch 27/30, Batch 200/938, Loss: 0.3402
Epoch 27/30, Batch 300/938, Loss: 0.1665
Epoch 27/30, Batch 400/938, Loss: 0.2710
Epoch 27/30, Batch 500/938, Loss: 0.2183
Epoch 27/30, Batch 600/938, Loss: 0.1031
Epoch 27/30, Batch 700/938, Loss: 0.3711
Epoch 27/30, Batch 800/938, Loss: 0.1156
Epoch 27/30, Batch 900/938, Loss: 0.2674
Epoch [27/30], Loss: 0.1798, Train Acc: 93.46%, Test Acc: 92.97%, LR: 0.000001
Epoch 28/30, Batch 0/938, Loss: 0.1500
Epoch 28/30, Batch 100/938, Loss: 0.1007
Epoch 28/30, Batch 200/938, Loss: 0.2193
Epoch 28/30, Batch 300/938, Loss: 0.1250
Epoch 28/30, Batch 400/938, Loss: 0.1485
Epoch 28/30, Batch 500/938, Loss: 0.1739
Epoch 28/30, Batch 600/938, Loss: 0.1797
Epoch 28/30, Batch 700/938, Loss: 0.2618
Epoch 28/30, Batch 800/938, Loss: 0.1787
Epoch 28/30, Batch 900/938, Loss: 0.3637
Epoch [28/30], Loss: 0.1789, Train Acc: 93.55%, Test Acc: 93.01%, LR: 0.000001
Epoch 29/30, Batch 0/938, Loss: 0.1609
Epoch 29/30, Batch 100/938, Loss: 0.1909
Epoch 29/30, Batch 200/938, Loss: 0.1768
Epoch 29/30, Batch 300/938, Loss: 0.1672
Epoch 29/30, Batch 400/938, Loss: 0.3386
Epoch 29/30, Batch 500/938, Loss: 0.2255
Epoch 29/30, Batch 600/938, Loss: 0.1839
Epoch 29/30, Batch 700/938, Loss: 0.1491
Epoch 29/30, Batch 800/938, Loss: 0.1458
Epoch 29/30, Batch 900/938, Loss: 0.1016
Epoch [29/30], Loss: 0.1796, Train Acc: 93.55%, Test Acc: 93.00%, LR: 0.000000
Epoch 30/30, Batch 0/938, Loss: 0.2638
Epoch 30/30, Batch 100/938, Loss: 0.2850
Epoch 30/30, Batch 200/938, Loss: 0.0970
Epoch 30/30, Batch 300/938, Loss: 0.1329
Epoch 30/30, Batch 400/938, Loss: 0.2984
Epoch 30/30, Batch 500/938, Loss: 0.1967
Epoch 30/30, Batch 600/938, Loss: 0.2134
Epoch 30/30, Batch 700/938, Loss: 0.1224
Epoch 30/30, Batch 800/938, Loss: 0.1732
Epoch 30/30, Batch 900/938, Loss: 0.2099
Epoch [30/30], Loss: 0.1778, Train Acc: 93.48%, Test Acc: 93.00%, LR: 0.000000
load best model, test accuracy: 93.13%
training time: 217.49s
best test accuracy: 93.13%
final test accuracy: 93.00%

============================================================
test config: ModifiedAlexNet_v3
============================================================
total params: 8,378,890
trainable params: 8,378,890
Epoch 1/30, Batch 0/469, Loss: 2.3019
Epoch 1/30, Batch 100/469, Loss: 0.8435
Epoch 1/30, Batch 200/469, Loss: 0.5660
Epoch 1/30, Batch 300/469, Loss: 0.8431
Epoch 1/30, Batch 400/469, Loss: 0.5066
Epoch [1/30], Loss: 0.8077, Train Acc: 69.62%, Test Acc: 80.85%, LR: 0.002000
Epoch 2/30, Batch 0/469, Loss: 0.4863
Epoch 2/30, Batch 100/469, Loss: 0.5008
Epoch 2/30, Batch 200/469, Loss: 0.5394
Epoch 2/30, Batch 300/469, Loss: 0.6222
Epoch 2/30, Batch 400/469, Loss: 0.4094
Epoch [2/30], Loss: 0.4960, Train Acc: 82.06%, Test Acc: 84.48%, LR: 0.002000
Epoch 3/30, Batch 0/469, Loss: 0.3635
Epoch 3/30, Batch 100/469, Loss: 0.3942
Epoch 3/30, Batch 200/469, Loss: 0.2669
Epoch 3/30, Batch 300/469, Loss: 0.3447
Epoch 3/30, Batch 400/469, Loss: 0.5286
Epoch [3/30], Loss: 0.4186, Train Acc: 84.95%, Test Acc: 87.20%, LR: 0.002000
Epoch 4/30, Batch 0/469, Loss: 0.3560
Epoch 4/30, Batch 100/469, Loss: 0.3489
Epoch 4/30, Batch 200/469, Loss: 0.4791
Epoch 4/30, Batch 300/469, Loss: 0.3798
Epoch 4/30, Batch 400/469, Loss: 0.3496
Epoch [4/30], Loss: 0.3902, Train Acc: 86.02%, Test Acc: 88.42%, LR: 0.002000
Epoch 5/30, Batch 0/469, Loss: 0.2948
Epoch 5/30, Batch 100/469, Loss: 0.3446
Epoch 5/30, Batch 200/469, Loss: 0.2461
Epoch 5/30, Batch 300/469, Loss: 0.4513
Epoch 5/30, Batch 400/469, Loss: 0.3942
Epoch [5/30], Loss: 0.3698, Train Acc: 86.78%, Test Acc: 88.55%, LR: 0.002000
Epoch 6/30, Batch 0/469, Loss: 0.3623
Epoch 6/30, Batch 100/469, Loss: 0.2821
Epoch 6/30, Batch 200/469, Loss: 0.3769
Epoch 6/30, Batch 300/469, Loss: 0.3249
Epoch 6/30, Batch 400/469, Loss: 0.3521
Epoch [6/30], Loss: 0.3583, Train Acc: 87.25%, Test Acc: 88.03%, LR: 0.002000
Epoch 7/30, Batch 0/469, Loss: 0.3285
Epoch 7/30, Batch 100/469, Loss: 0.2690
Epoch 7/30, Batch 200/469, Loss: 0.3822
Epoch 7/30, Batch 300/469, Loss: 0.3704
Epoch 7/30, Batch 400/469, Loss: 0.2541
Epoch [7/30], Loss: 0.3472, Train Acc: 87.72%, Test Acc: 89.30%, LR: 0.002000
Epoch 8/30, Batch 0/469, Loss: 0.3097
Epoch 8/30, Batch 100/469, Loss: 0.4423
Epoch 8/30, Batch 200/469, Loss: 0.2555
Epoch 8/30, Batch 300/469, Loss: 0.3827
Epoch 8/30, Batch 400/469, Loss: 0.2429
Epoch [8/30], Loss: 0.2865, Train Acc: 89.72%, Test Acc: 89.99%, LR: 0.000200
Epoch 9/30, Batch 0/469, Loss: 0.3015
Epoch 9/30, Batch 100/469, Loss: 0.2270
Epoch 9/30, Batch 200/469, Loss: 0.2683
Epoch 9/30, Batch 300/469, Loss: 0.2781
Epoch 9/30, Batch 400/469, Loss: 0.3125
Epoch [9/30], Loss: 0.2723, Train Acc: 90.16%, Test Acc: 90.16%, LR: 0.000200
Epoch 10/30, Batch 0/469, Loss: 0.3393
Epoch 10/30, Batch 100/469, Loss: 0.2865
Epoch 10/30, Batch 200/469, Loss: 0.3694
Epoch 10/30, Batch 300/469, Loss: 0.2022
Epoch 10/30, Batch 400/469, Loss: 0.2193
Epoch [10/30], Loss: 0.2691, Train Acc: 90.29%, Test Acc: 90.33%, LR: 0.000200
Epoch 11/30, Batch 0/469, Loss: 0.2710
Epoch 11/30, Batch 100/469, Loss: 0.2724
Epoch 11/30, Batch 200/469, Loss: 0.3046
Epoch 11/30, Batch 300/469, Loss: 0.1777
Epoch 11/30, Batch 400/469, Loss: 0.1555
Epoch [11/30], Loss: 0.2681, Train Acc: 90.42%, Test Acc: 90.49%, LR: 0.000200
Epoch 12/30, Batch 0/469, Loss: 0.2823
Epoch 12/30, Batch 100/469, Loss: 0.2870
Epoch 12/30, Batch 200/469, Loss: 0.2743
Epoch 12/30, Batch 300/469, Loss: 0.2652
Epoch 12/30, Batch 400/469, Loss: 0.2050
Epoch [12/30], Loss: 0.2655, Train Acc: 90.38%, Test Acc: 90.70%, LR: 0.000200
Epoch 13/30, Batch 0/469, Loss: 0.2252
Epoch 13/30, Batch 100/469, Loss: 0.2830
Epoch 13/30, Batch 200/469, Loss: 0.3538
Epoch 13/30, Batch 300/469, Loss: 0.3432
Epoch 13/30, Batch 400/469, Loss: 0.4293
Epoch [13/30], Loss: 0.2629, Train Acc: 90.54%, Test Acc: 90.77%, LR: 0.000200
Epoch 14/30, Batch 0/469, Loss: 0.3097
Epoch 14/30, Batch 100/469, Loss: 0.2128
Epoch 14/30, Batch 200/469, Loss: 0.2185
Epoch 14/30, Batch 300/469, Loss: 0.1638
Epoch 14/30, Batch 400/469, Loss: 0.2754
Epoch [14/30], Loss: 0.2611, Train Acc: 90.53%, Test Acc: 90.58%, LR: 0.000200
Epoch 15/30, Batch 0/469, Loss: 0.2137
Epoch 15/30, Batch 100/469, Loss: 0.2680
Epoch 15/30, Batch 200/469, Loss: 0.3501
Epoch 15/30, Batch 300/469, Loss: 0.2524
Epoch 15/30, Batch 400/469, Loss: 0.2201
Epoch [15/30], Loss: 0.2495, Train Acc: 90.97%, Test Acc: 91.04%, LR: 0.000020
Epoch 16/30, Batch 0/469, Loss: 0.2266
Epoch 16/30, Batch 100/469, Loss: 0.2882
Epoch 16/30, Batch 200/469, Loss: 0.2573
Epoch 16/30, Batch 300/469, Loss: 0.3294
Epoch 16/30, Batch 400/469, Loss: 0.1485
Epoch [16/30], Loss: 0.2507, Train Acc: 91.03%, Test Acc: 91.00%, LR: 0.000020
Epoch 17/30, Batch 0/469, Loss: 0.2742
Epoch 17/30, Batch 100/469, Loss: 0.2692
Epoch 17/30, Batch 200/469, Loss: 0.1880
Epoch 17/30, Batch 300/469, Loss: 0.2257
Epoch 17/30, Batch 400/469, Loss: 0.2494
Epoch [17/30], Loss: 0.2480, Train Acc: 91.06%, Test Acc: 90.95%, LR: 0.000020
Epoch 18/30, Batch 0/469, Loss: 0.2683
Epoch 18/30, Batch 100/469, Loss: 0.3458
Epoch 18/30, Batch 200/469, Loss: 0.2788
Epoch 18/30, Batch 300/469, Loss: 0.2515
Epoch 18/30, Batch 400/469, Loss: 0.2530
Epoch [18/30], Loss: 0.2463, Train Acc: 91.19%, Test Acc: 91.05%, LR: 0.000020
Epoch 19/30, Batch 0/469, Loss: 0.1859
Epoch 19/30, Batch 100/469, Loss: 0.1845
Epoch 19/30, Batch 200/469, Loss: 0.2777
Epoch 19/30, Batch 300/469, Loss: 0.1765
Epoch 19/30, Batch 400/469, Loss: 0.2691
Epoch [19/30], Loss: 0.2474, Train Acc: 91.12%, Test Acc: 90.91%, LR: 0.000020
Epoch 20/30, Batch 0/469, Loss: 0.3504
Epoch 20/30, Batch 100/469, Loss: 0.1458
Epoch 20/30, Batch 200/469, Loss: 0.4078
Epoch 20/30, Batch 300/469, Loss: 0.2587
Epoch 20/30, Batch 400/469, Loss: 0.3099
Epoch [20/30], Loss: 0.2456, Train Acc: 91.17%, Test Acc: 91.03%, LR: 0.000020
Epoch 21/30, Batch 0/469, Loss: 0.3048
Epoch 21/30, Batch 100/469, Loss: 0.2918
Epoch 21/30, Batch 200/469, Loss: 0.2501
Epoch 21/30, Batch 300/469, Loss: 0.1594
Epoch 21/30, Batch 400/469, Loss: 0.2252
Epoch [21/30], Loss: 0.2449, Train Acc: 91.12%, Test Acc: 90.96%, LR: 0.000020
Epoch 22/30, Batch 0/469, Loss: 0.1844
Epoch 22/30, Batch 100/469, Loss: 0.3484
Epoch 22/30, Batch 200/469, Loss: 0.2870
Epoch 22/30, Batch 300/469, Loss: 0.2969
Epoch 22/30, Batch 400/469, Loss: 0.1712
Epoch [22/30], Loss: 0.2458, Train Acc: 91.12%, Test Acc: 90.99%, LR: 0.000002
Epoch 23/30, Batch 0/469, Loss: 0.1736
Epoch 23/30, Batch 100/469, Loss: 0.1553
Epoch 23/30, Batch 200/469, Loss: 0.2241
Epoch 23/30, Batch 300/469, Loss: 0.2629
Epoch 23/30, Batch 400/469, Loss: 0.2421
Epoch [23/30], Loss: 0.2432, Train Acc: 91.28%, Test Acc: 91.09%, LR: 0.000002
Epoch 24/30, Batch 0/469, Loss: 0.1978
Epoch 24/30, Batch 100/469, Loss: 0.3327
Epoch 24/30, Batch 200/469, Loss: 0.2460
Epoch 24/30, Batch 300/469, Loss: 0.2594
Epoch 24/30, Batch 400/469, Loss: 0.2055
Epoch [24/30], Loss: 0.2430, Train Acc: 91.30%, Test Acc: 91.06%, LR: 0.000002
Epoch 25/30, Batch 0/469, Loss: 0.2759
Epoch 25/30, Batch 100/469, Loss: 0.3102
Epoch 25/30, Batch 200/469, Loss: 0.2643
Epoch 25/30, Batch 300/469, Loss: 0.3129
Epoch 25/30, Batch 400/469, Loss: 0.2536
Epoch [25/30], Loss: 0.2429, Train Acc: 91.23%, Test Acc: 91.05%, LR: 0.000002
Epoch 26/30, Batch 0/469, Loss: 0.2857
Epoch 26/30, Batch 100/469, Loss: 0.2833
Epoch 26/30, Batch 200/469, Loss: 0.1903
Epoch 26/30, Batch 300/469, Loss: 0.2197
Epoch 26/30, Batch 400/469, Loss: 0.3400
Epoch [26/30], Loss: 0.2436, Train Acc: 91.17%, Test Acc: 91.07%, LR: 0.000002
Epoch 27/30, Batch 0/469, Loss: 0.1909
Epoch 27/30, Batch 100/469, Loss: 0.2239
Epoch 27/30, Batch 200/469, Loss: 0.3471
Epoch 27/30, Batch 300/469, Loss: 0.3418
Epoch 27/30, Batch 400/469, Loss: 0.2586
Epoch [27/30], Loss: 0.2426, Train Acc: 91.28%, Test Acc: 91.05%, LR: 0.000002
Epoch 28/30, Batch 0/469, Loss: 0.2451
Epoch 28/30, Batch 100/469, Loss: 0.1673
Epoch 28/30, Batch 200/469, Loss: 0.3643
Epoch 28/30, Batch 300/469, Loss: 0.3086
Epoch 28/30, Batch 400/469, Loss: 0.2367
Epoch [28/30], Loss: 0.2436, Train Acc: 91.17%, Test Acc: 91.05%, LR: 0.000002
Epoch 29/30, Batch 0/469, Loss: 0.1572
Epoch 29/30, Batch 100/469, Loss: 0.1866
Epoch 29/30, Batch 200/469, Loss: 0.1899
Epoch 29/30, Batch 300/469, Loss: 0.2597
Epoch 29/30, Batch 400/469, Loss: 0.2451
Epoch [29/30], Loss: 0.2447, Train Acc: 91.12%, Test Acc: 91.05%, LR: 0.000000
Epoch 30/30, Batch 0/469, Loss: 0.1272
Epoch 30/30, Batch 100/469, Loss: 0.2232
Epoch 30/30, Batch 200/469, Loss: 0.2492
Epoch 30/30, Batch 300/469, Loss: 0.2467
Epoch 30/30, Batch 400/469, Loss: 0.1772
Epoch [30/30], Loss: 0.2441, Train Acc: 91.17%, Test Acc: 91.05%, LR: 0.000000
load best model, test accuracy: 91.09%
training time: 185.73s
best test accuracy: 91.09%
final test accuracy: 91.05%