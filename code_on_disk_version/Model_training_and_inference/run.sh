# python main.py recognition -c config/st_gcn/handwash/jh/train_1.yaml &
# python main.py recognition -c config/st_gcn/handwash/jh/train_2.yaml &
# python main.py recognition -c config/st_gcn/handwash/jh/train_3.yaml &

for i in {0..6}; do
    python main.py recognition \
    -c config/st_gcn/handwash/train.yaml \
    --step ${i} \
    > log_step_${i} 2>&1
done