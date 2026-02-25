from pipelines.run_runner import pipeline
from pipelines.train import training_pipeline,valid_pipeline
from pipelines.statistic import get_params

def run_statistic_pipeline():
    config_path = "configs/statisticConfig.yaml"
    get_params(config_path)

def run_runner_pipeline():
    config_path = "configs/optimizerConfig.yaml"
    is_processed = False
    processed_timestamp = "20260218_035503"
    # idxs = [760,1903,2207,479,596,634]
    idxs = CHANGE_IDXS
    pipeline(config_path, idxs, is_processed, processed_timestamp)

def run_train_pipeline():
    config_path = "configs/trainConfig.yaml"
    training_pipeline(config_path)

def run_valid_pipeline():
    config_path = "configs/trainConfig.yaml"
    model_checkpoint_path = "checkpoints/attunet/20260212_045750/best_model.pth"
    # model_checkpoint_path = "checkpoints/unet/20260212_155606/best_model.pth"
    # model_checkpoint_path = "checkpoints/segformer/20260213_062110/best_model.pth"
    valid_pipeline(config_path, model_checkpoint_path)

CHANGE_IDXS = [355, 356, 417, 418, 451, 452, 477, 478, 479, 480, 513, 514, 515, 516, 539, 540, 541, 570, 571, 573, 574, 575, 576, 577, 578, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 632, 633, 634, 635, 636, 637, 638, 639, 640, 656, 657, 658, 659, 660, 661, 662, 663, 664, 694, 695, 696, 697, 698, 699, 700, 701, 702, 723, 724, 758, 759, 760, 761, 762, 763, 765, 766, 767, 821, 822, 823, 824, 825, 827, 828, 829, 830, 885, 886, 887, 889, 890, 891, 892, 947, 948, 949, 951, 952, 953, 1009, 1010, 1013, 1014, 1015, 1071, 1072, 1235, 1236, 1237, 1283, 1284, 1296, 1297, 1298, 1299, 1345, 1346, 1355, 1356, 1358, 1359, 1360, 1390, 1391, 1402, 1403, 1417, 1418, 1452, 1453, 1454, 1455, 1456, 1457, 1464, 1465, 1466, 1514, 1515, 1516, 1517, 1518, 1519, 1526, 1527, 1528, 1579, 1580, 1581, 1582, 1583, 1644, 1645, 1651, 1652, 1657, 1658, 1659, 1661, 1662, 1663, 1664, 1706, 1707, 1713, 1714, 1715, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726, 1729, 1730, 1731, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1783, 1784, 1785, 1786, 1787, 1788, 1790, 1791, 1792, 1793, 1794, 1795, 1836, 1837, 1838, 1839, 1840, 1841, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1852, 1853, 1854, 1855, 1856, 1857, 1874, 1875, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1915, 1916, 1917, 1918, 1919, 1936, 1937, 1960, 1961, 1962, 1963, 1964, 1978, 1979, 1980, 2083, 2084, 2138, 2139, 2140, 2142, 2143, 2144, 2145, 2146, 2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2261, 2262, 2263, 2264, 2265, 2266, 2267, 2268, 2324, 2325, 2326, 2327, 2328, 2329, 2330, 2377, 2378, 2387, 2388, 2389, 2390, 2391, 2392, 2439, 2440, 2451, 2452, 2501, 2502, 2563, 2564, 2625, 2626]

if __name__ == "__main__":
    # run_train_pipeline()
    # run_valid_pipeline()
    run_runner_pipeline()
    # create_split_pipeline()