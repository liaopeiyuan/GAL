CUDA_VISIBLE_DEVICES=0 sh ./config/FB15k-237/ablation/baseline_conve_1000_1.sh &
CUDA_VISIBLE_DEVICES=1 sh ./config/FB15k-237/ablation/baseline_conve_2000_1.sh &
CUDA_VISIBLE_DEVICES=2 sh ./config/FB15k-237/ablation/baseline_conve_3000_1.sh &
CUDA_VISIBLE_DEVICES=3 sh ./config/FB15k-237/ablation/baseline_conve_4000_1.sh &
CUDA_VISIBLE_DEVICES=4 sh ./config/FB15k-237/ablation/adv_gnn_conve_1000_1.sh &
CUDA_VISIBLE_DEVICES=5 sh ./config/FB15k-237/ablation/adv_gnn_conve_2000_1.sh &
CUDA_VISIBLE_DEVICES=6 sh ./config/FB15k-237/ablation/adv_gnn_conve_3000_1.sh &
CUDA_VISIBLE_DEVICES=7 sh ./config/FB15k-237/ablation/adv_gnn_conve_4000_1.sh &

