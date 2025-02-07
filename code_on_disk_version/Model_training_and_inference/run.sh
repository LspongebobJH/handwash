# python main.py recognition -c config/st_gcn/handwash/jh/train_1.yaml &
# python main.py recognition -c config/st_gcn/handwash/jh/train_2.yaml &
# python main.py recognition -c config/st_gcn/handwash/jh/train_3.yaml &

# for i in {1..6}; do
#     python main.py recognition \
#     -c config/st_gcn/handwash/train.yaml \
#     --step ${i}
# done

for i in {0..6}; do
    touch ./config/st_gcn/handwash/binary/inference_${i}.yaml
done