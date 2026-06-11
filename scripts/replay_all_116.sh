#!/bin/bash
# Replay 1 successful HP+PDM grasp per object
# Scene loads → waits for Enter → executes grasp → next object

PROJ=/home/lyh/Project/Affordance2Grasp
cd "$PROJ"

echo "HP+PDM Grasp Replay: 97/116 objects with success"
echo "Each object: scene loads → wait for Enter → grasp executes → next"
echo ""

echo "=== [1/97] A01001 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A01001 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A01001/A01001_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.016813,0.011671 --episode-id A01001_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [2/97] A01002 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A01002 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A01002/A01002_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.018280,0.010667 --episode-id A01002_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [3/97] A01005 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A01005 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A01005/A01005_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.002730,0.020947 --episode-id A01005_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [4/97] A01006 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A01006 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A01006/A01006_yaw000_pool_grasp.hdf5 --candidate-index 1 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.031255,-0.013183 --episode-id A01006_a2g_pdm_yaw000_t001 --wait-before-grasp --loud

echo "=== [5/97] A01008 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A01008 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A01008/A01008_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.021634,0.042316 --episode-id A01008_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [6/97] A01009 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A01009 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A01009/A01009_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.042027,0.030872 --episode-id A01009_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [7/97] A01010 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A01010 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A01010/A01010_yaw000_pool_grasp.hdf5 --candidate-index 2 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.006948,-0.030414 --episode-id A01010_a2g_pdm_yaw000_t002 --wait-before-grasp --loud

echo "=== [8/97] A01023 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A01023 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A01023/A01023_yaw000_pool_grasp.hdf5 --candidate-index 1 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.034888,-0.039513 --episode-id A01023_a2g_pdm_yaw000_t001 --wait-before-grasp --loud

echo "=== [9/97] A01026 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A01026 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A01026/A01026_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.021525,-0.022420 --episode-id A01026_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [10/97] A01027 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A01027 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A01027/A01027_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.035137,-0.003513 --episode-id A01027_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [11/97] A02011 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A02011 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A02011/A02011_yaw090_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.039209,0.046575 --episode-id A02011_a2g_pdm_yaw090_t000 --wait-before-grasp --loud

echo "=== [12/97] A02012 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A02012 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A02012/A02012_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.025432,0.041906 --episode-id A02012_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [13/97] A02014 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A02014 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A02014/A02014_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.026760,0.023204 --episode-id A02014_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [14/97] A02015 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A02015 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A02015/A02015_yaw000_pool_grasp.hdf5 --candidate-index 2 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.002308,0.021267 --episode-id A02015_a2g_pdm_yaw000_t002 --wait-before-grasp --loud

echo "=== [15/97] A02018 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A02018 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A02018/A02018_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.045218,-0.004348 --episode-id A02018_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [16/97] A02021 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A02021 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A02021/A02021_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.017551,0.029860 --episode-id A02021_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [17/97] A02028 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A02028 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A02028/A02028_yaw000_pool_grasp.hdf5 --candidate-index 1 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.045536,0.049337 --episode-id A02028_a2g_pdm_yaw000_t001 --wait-before-grasp --loud

echo "=== [18/97] A02029 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A02029 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A02029/A02029_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.026618,0.025672 --episode-id A02029_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [19/97] A02030 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A02030 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A02030/A02030_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.004850,-0.024990 --episode-id A02030_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [20/97] A02031 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A02031 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A02031/A02031_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.045963,-0.002706 --episode-id A02031_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [21/97] A15015 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A15015 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A15015/A15015_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.048919,-0.034336 --episode-id A15015_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [22/97] A15027 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A15027 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A15027/A15027_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.006859,-0.010592 --episode-id A15027_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [23/97] A16012 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A16012 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A16012/A16012_yaw000_pool_grasp.hdf5 --candidate-index 3 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.033636,0.032265 --episode-id A16012_a2g_pdm_yaw000_t003 --wait-before-grasp --loud

echo "=== [24/97] A16013 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A16013 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A16013/A16013_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.012179,-0.027437 --episode-id A16013_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [25/97] A16026 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id A16026 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/A16026/A16026_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.027483,0.032286 --episode-id A16026_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [26/97] C03001 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id C03001 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/C03001/C03001_yaw000_pool_grasp.hdf5 --candidate-index 5 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.035106,0.040650 --episode-id C03001_a2g_pdm_yaw000_t005 --wait-before-grasp --loud

echo "=== [27/97] C14001 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id C14001 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/C14001/C14001_yaw000_pool_grasp.hdf5 --candidate-index 2 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.010791,-0.021404 --episode-id C14001_a2g_pdm_yaw000_t002 --wait-before-grasp --loud

echo "=== [28/97] C15001 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id C15001 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/C15001/C15001_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.000628,0.045474 --episode-id C15001_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [29/97] C28001 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id C28001 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/C28001/C28001_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.007880,-0.027241 --episode-id C28001_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [30/97] C40001 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id C40001 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/C40001/C40001_yaw000_pool_grasp.hdf5 --candidate-index 1 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.004541,-0.045358 --episode-id C40001_a2g_pdm_yaw000_t001 --wait-before-grasp --loud

echo "=== [31/97] C50001 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id C50001 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/C50001/C50001_yaw000_pool_grasp.hdf5 --candidate-index 3 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.030295,-0.023263 --episode-id C50001_a2g_pdm_yaw000_t003 --wait-before-grasp --loud

echo "=== [32/97] C52001 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id C52001 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/C52001/C52001_yaw000_pool_grasp.hdf5 --candidate-index 2 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.008677,-0.012464 --episode-id C52001_a2g_pdm_yaw000_t002 --wait-before-grasp --loud

echo "=== [33/97] C90001 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id C90001 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/C90001/C90001_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.014433,-0.032879 --episode-id C90001_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [34/97] O01000 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id O01000 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/O01000/O01000_yaw000_pool_grasp.hdf5 --candidate-index 4 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.023796,0.035957 --episode-id O01000_a2g_pdm_yaw000_t004 --wait-before-grasp --loud

echo "=== [35/97] O02001 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id O02001 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/O02001/O02001_yaw000_pool_grasp.hdf5 --candidate-index 7 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.036356,0.038305 --episode-id O02001_a2g_pdm_yaw000_t007 --wait-before-grasp --loud

echo "=== [36/97] O03001 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id O03001 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/O03001/O03001_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.012910,-0.018613 --episode-id O03001_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [37/97] O03002 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id O03002 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/O03002/O03002_yaw000_pool_grasp.hdf5 --candidate-index 1 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.043037,0.024694 --episode-id O03002_a2g_pdm_yaw000_t001 --wait-before-grasp --loud

echo "=== [38/97] O03003 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id O03003 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/O03003/O03003_yaw000_pool_grasp.hdf5 --candidate-index 1 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.010474,-0.036862 --episode-id O03003_a2g_pdm_yaw000_t001 --wait-before-grasp --loud

echo "=== [39/97] O21001 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id O21001 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/O21001/O21001_yaw000_pool_grasp.hdf5 --candidate-index 6 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.026850,0.039170 --episode-id O21001_a2g_pdm_yaw000_t006 --wait-before-grasp --loud

echo "=== [40/97] O36001 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id O36001 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/O36001/O36001_yaw000_pool_grasp.hdf5 --candidate-index 5 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.029116,-0.033066 --episode-id O36001_a2g_pdm_yaw000_t005 --wait-before-grasp --loud

echo "=== [41/97] O36002 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id O36002 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/O36002/O36002_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.005178,0.040831 --episode-id O36002_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [42/97] O50001 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id O50001 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/O50001/O50001_yaw000_pool_grasp.hdf5 --candidate-index 2 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.005242,0.048446 --episode-id O50001_a2g_pdm_yaw000_t002 --wait-before-grasp --loud

echo "=== [43/97] S10005 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id S10005 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/S10005/S10005_yaw000_pool_grasp.hdf5 --candidate-index 3 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.000662,0.020262 --episode-id S10005_a2g_pdm_yaw000_t003 --wait-before-grasp --loud

echo "=== [44/97] S10008 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id S10008 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/S10008/S10008_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.018012,0.019349 --episode-id S10008_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [45/97] S10010 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id S10010 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/S10010/S10010_yaw000_pool_grasp.hdf5 --candidate-index 1 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.020091,0.012769 --episode-id S10010_a2g_pdm_yaw000_t001 --wait-before-grasp --loud

echo "=== [46/97] S10011 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id S10011 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/S10011/S10011_yaw000_pool_grasp.hdf5 --candidate-index 1 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.047853,-0.020123 --episode-id S10011_a2g_pdm_yaw000_t001 --wait-before-grasp --loud

echo "=== [47/97] S10013 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id S10013 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/S10013/S10013_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.024489,-0.028850 --episode-id S10013_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [48/97] S10014 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id S10014 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/S10014/S10014_yaw000_pool_grasp.hdf5 --candidate-index 2 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.016768,-0.048925 --episode-id S10014_a2g_pdm_yaw000_t002 --wait-before-grasp --loud

echo "=== [49/97] S10017 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id S10017 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/S10017/S10017_yaw000_pool_grasp.hdf5 --candidate-index 1 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.033495,0.001564 --episode-id S10017_a2g_pdm_yaw000_t001 --wait-before-grasp --loud

echo "=== [50/97] S10018 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id S10018 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/S10018/S10018_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.013302,0.010907 --episode-id S10018_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [51/97] S10020 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id S10020 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/S10020/S10020_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.006352,-0.041177 --episode-id S10020_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [52/97] S10022 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id S10022 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/S10022/S10022_yaw000_pool_grasp.hdf5 --candidate-index 1 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.001224,-0.033593 --episode-id S10022_a2g_pdm_yaw000_t001 --wait-before-grasp --loud

echo "=== [53/97] S15004 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id S15004 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/S15004/S15004_yaw000_pool_grasp.hdf5 --candidate-index 1 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.023457,-0.037643 --episode-id S15004_a2g_pdm_yaw000_t001 --wait-before-grasp --loud

echo "=== [54/97] S16001 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id S16001 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/S16001/S16001_yaw000_pool_grasp.hdf5 --candidate-index 1 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.001592,-0.014528 --episode-id S16001_a2g_pdm_yaw000_t001 --wait-before-grasp --loud

echo "=== [55/97] S16002 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id S16002 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/S16002/S16002_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.032717,0.036593 --episode-id S16002_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [56/97] S16003 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id S16003 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/S16003/S16003_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.040546,-0.030924 --episode-id S16003_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [57/97] S16005 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id S16005 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/S16005/S16005_yaw000_pool_grasp.hdf5 --candidate-index 6 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.038879,0.032341 --episode-id S16005_a2g_pdm_yaw000_t006 --wait-before-grasp --loud

echo "=== [58/97] S20001 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id S20001 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/S20001/S20001_yaw000_pool_grasp.hdf5 --candidate-index 4 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.020544,-0.010430 --episode-id S20001_a2g_pdm_yaw000_t004 --wait-before-grasp --loud

echo "=== [59/97] S20021 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id S20021 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/S20021/S20021_yaw180_pool_grasp.hdf5 --candidate-index 2 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.020851,-0.015690 --episode-id S20021_a2g_pdm_yaw180_t002 --wait-before-grasp --loud

echo "=== [60/97] S20022 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id S20022 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/S20022/S20022_yaw000_pool_grasp.hdf5 --candidate-index 8 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.036445,0.021060 --episode-id S20022_a2g_pdm_yaw000_t008 --wait-before-grasp --loud

echo "=== [61/97] Y03006 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id Y03006 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/Y03006/Y03006_yaw000_pool_grasp.hdf5 --candidate-index 2 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.023758,0.044689 --episode-id Y03006_a2g_pdm_yaw000_t002 --wait-before-grasp --loud

echo "=== [62/97] Y03021 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id Y03021 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/Y03021/Y03021_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.027236,0.004048 --episode-id Y03021_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [63/97] Y27035 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id Y27035 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/Y27035/Y27035_yaw000_pool_grasp.hdf5 --candidate-index 1 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.029534,0.026481 --episode-id Y27035_a2g_pdm_yaw000_t001 --wait-before-grasp --loud

echo "=== [64/97] Y29040 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id Y29040 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/Y29040/Y29040_yaw000_pool_grasp.hdf5 --candidate-index 7 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.026897,-0.015294 --episode-id Y29040_a2g_pdm_yaw000_t007 --wait-before-grasp --loud

echo "=== [65/97] Y35037 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id Y35037 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/Y35037/Y35037_yaw000_pool_grasp.hdf5 --candidate-index 2 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.015306,0.042581 --episode-id Y35037_a2g_pdm_yaw000_t002 --wait-before-grasp --loud

echo "=== [66/97] unseen_000 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id unseen_000 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/unseen_000/unseen_000_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.035297,0.017509 --episode-id unseen_000_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [67/97] unseen_001 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id unseen_001 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/unseen_001/unseen_001_yaw000_pool_grasp.hdf5 --candidate-index 1 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.037067,-0.035347 --episode-id unseen_001_a2g_pdm_yaw000_t001 --wait-before-grasp --loud

echo "=== [68/97] unseen_002 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id unseen_002 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/unseen_002/unseen_002_yaw090_pool_grasp.hdf5 --candidate-index 9 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.007205,-0.004636 --episode-id unseen_002_a2g_pdm_yaw090_t009 --wait-before-grasp --loud

echo "=== [69/97] unseen_004 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id unseen_004 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/unseen_004/unseen_004_yaw000_pool_grasp.hdf5 --candidate-index 1 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.036397,0.025058 --episode-id unseen_004_a2g_pdm_yaw000_t001 --wait-before-grasp --loud

echo "=== [70/97] unseen_005 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id unseen_005 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/unseen_005/unseen_005_yaw000_pool_grasp.hdf5 --candidate-index 5 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.003802,-0.041147 --episode-id unseen_005_a2g_pdm_yaw000_t005 --wait-before-grasp --loud

echo "=== [71/97] unseen_006 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id unseen_006 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/unseen_006/unseen_006_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.001155,0.029169 --episode-id unseen_006_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [72/97] unseen_007 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id unseen_007 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/unseen_007/unseen_007_yaw000_pool_grasp.hdf5 --candidate-index 2 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.042759,-0.031908 --episode-id unseen_007_a2g_pdm_yaw000_t002 --wait-before-grasp --loud

echo "=== [73/97] unseen_008 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id unseen_008 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/unseen_008/unseen_008_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.017767,-0.047737 --episode-id unseen_008_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [74/97] unseen_009 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id unseen_009 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/unseen_009/unseen_009_yaw090_pool_grasp.hdf5 --candidate-index 7 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.041270,-0.004347 --episode-id unseen_009_a2g_pdm_yaw090_t007 --wait-before-grasp --loud

echo "=== [75/97] unseen_010 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id unseen_010 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/unseen_010/unseen_010_yaw000_pool_grasp.hdf5 --candidate-index 7 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.033457,0.016093 --episode-id unseen_010_a2g_pdm_yaw000_t007 --wait-before-grasp --loud

echo "=== [76/97] unseen_011 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id unseen_011 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/unseen_011/unseen_011_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.018742,-0.044547 --episode-id unseen_011_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [77/97] unseen_014 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id unseen_014 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/unseen_014/unseen_014_yaw000_pool_grasp.hdf5 --candidate-index 8 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.040193,0.014370 --episode-id unseen_014_a2g_pdm_yaw000_t008 --wait-before-grasp --loud

echo "=== [78/97] unseen_015 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id unseen_015 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/unseen_015/unseen_015_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.024950,-0.035751 --episode-id unseen_015_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [79/97] unseen_016 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id unseen_016 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/unseen_016/unseen_016_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.039249,-0.016823 --episode-id unseen_016_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [80/97] unseen_019 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id unseen_019 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/unseen_019/unseen_019_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.021583,-0.024831 --episode-id unseen_019_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [81/97] unseen_020 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id unseen_020 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/unseen_020/unseen_020_yaw090_pool_grasp.hdf5 --candidate-index 9 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.008491,-0.012430 --episode-id unseen_020_a2g_pdm_yaw090_t009 --wait-before-grasp --loud

echo "=== [82/97] unseen_022 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id unseen_022 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/unseen_022/unseen_022_yaw000_pool_grasp.hdf5 --candidate-index 2 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.021781,0.033905 --episode-id unseen_022_a2g_pdm_yaw000_t002 --wait-before-grasp --loud

echo "=== [83/97] unseen_024 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id unseen_024 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/unseen_024/unseen_024_yaw000_pool_grasp.hdf5 --candidate-index 5 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.011646,-0.048703 --episode-id unseen_024_a2g_pdm_yaw000_t005 --wait-before-grasp --loud

echo "=== [84/97] unseen_025 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id unseen_025 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/unseen_025/unseen_025_yaw000_pool_grasp.hdf5 --candidate-index 9 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.005079,-0.037566 --episode-id unseen_025_a2g_pdm_yaw000_t009 --wait-before-grasp --loud

echo "=== [85/97] unseen_026 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id unseen_026 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/unseen_026/unseen_026_yaw000_pool_grasp.hdf5 --candidate-index 1 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.007597,-0.043191 --episode-id unseen_026_a2g_pdm_yaw000_t001 --wait-before-grasp --loud

echo "=== [86/97] unseen_027 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id unseen_027 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/unseen_027/unseen_027_yaw000_pool_grasp.hdf5 --candidate-index 5 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.042784,0.029298 --episode-id unseen_027_a2g_pdm_yaw000_t005 --wait-before-grasp --loud

echo "=== [87/97] unseen_028 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id unseen_028 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/unseen_028/unseen_028_yaw090_pool_grasp.hdf5 --candidate-index 7 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.004960,0.002972 --episode-id unseen_028_a2g_pdm_yaw090_t007 --wait-before-grasp --loud

echo "=== [88/97] ycb_dex_03 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id ycb_dex_03 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/ycb_dex_03/ycb_dex_03_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.008021,-0.017702 --episode-id ycb_dex_03_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [89/97] ycb_dex_04 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id ycb_dex_04 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/ycb_dex_04/ycb_dex_04_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.014189,0.000861 --episode-id ycb_dex_04_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [90/97] ycb_dex_05 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id ycb_dex_05 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/ycb_dex_05/ycb_dex_05_yaw000_pool_grasp.hdf5 --candidate-index 1 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.006322,0.021652 --episode-id ycb_dex_05_a2g_pdm_yaw000_t001 --wait-before-grasp --loud

echo "=== [91/97] ycb_dex_07 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id ycb_dex_07 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/ycb_dex_07/ycb_dex_07_yaw090_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.035892,0.027283 --episode-id ycb_dex_07_a2g_pdm_yaw090_t000 --wait-before-grasp --loud

echo "=== [92/97] ycb_dex_08 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id ycb_dex_08 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/ycb_dex_08/ycb_dex_08_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.034402,0.012452 --episode-id ycb_dex_08_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [93/97] ycb_dex_12 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id ycb_dex_12 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/ycb_dex_12/ycb_dex_12_yaw000_pool_grasp.hdf5 --candidate-index 6 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.023249,0.016581 --episode-id ycb_dex_12_a2g_pdm_yaw000_t006 --wait-before-grasp --loud

echo "=== [94/97] ycb_dex_14 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id ycb_dex_14 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/ycb_dex_14/ycb_dex_14_yaw000_pool_grasp.hdf5 --candidate-index 6 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.049534,-0.028669 --episode-id ycb_dex_14_a2g_pdm_yaw000_t006 --wait-before-grasp --loud

echo "=== [95/97] ycb_dex_15 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id ycb_dex_15 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/ycb_dex_15/ycb_dex_15_yaw000_pool_grasp.hdf5 --candidate-index 1 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset 0.037762,-0.030533 --episode-id ycb_dex_15_a2g_pdm_yaw000_t001 --wait-before-grasp --loud

echo "=== [96/97] ycb_dex_17 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id ycb_dex_17 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/ycb_dex_17/ycb_dex_17_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.021013,0.011198 --episode-id ycb_dex_17_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "=== [97/97] ycb_dex_18 yaw=0 ==="
/home/lyh/isaac-sim-5.0/python.sh evaluation/eval_single.py --obj-id ycb_dex_18 --candidate-hdf5 /home/lyh/Project/Affordance2Grasp/output/evaluation/hp_pdm_yaw4x10_random_xy_seed42/candidates/ycb_dex_18/ycb_dex_18_yaw000_pool_grasp.hdf5 --candidate-index 0 --selection index --z-yaw-deg 0 --random-obj-xy --obj-xy-offset -0.015578,-0.036450 --episode-id ycb_dex_18_a2g_pdm_yaw000_t000 --wait-before-grasp --loud

echo "Done! 97 objects replayed."
