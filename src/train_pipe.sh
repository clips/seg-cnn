#!/usr/bin/env bash

#check scale_fac in datagen.py is set to 100
#echo "___ trp:"
#cat="trp"
#python /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/src/datagen.py ${cat} # because of different settings for TrP, TeP and PP
#THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_semrel.py -static -word2vec -img_w200 -l1_nhu100 -pad7 -$cat -n_runs20 -n_train10000000 > ../result/semclass_${cat}

#echo "___ tep:"
#cat="tep"
#python /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/src/datagen.py ${cat} # because of different settings for TrP, TeP and PP
#THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_semrel.py -static -word2vec -img_w500 -l1_nhu150 -pad4 -$cat -n_runs20 -n_train10000000 > ../result/semclass_${cat}

#echo "___ pp:"
#cat="pp"
#python /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/src/datagen.py ${cat} # because of different settings for TrP, TeP and PP
#THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_semrel.py -static -word2vec -img_w400 -l1_nhu100 -pad10 -$cat -n_runs20 -n_train10000000 > ../result/semclass_${cat}




#set scale_fac in datagen.py to 0 to ignore external features
#echo "___ trp:"
#cat="trp"
#python /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/src/datagen.py ${cat} # because of different settings for TrP, TeP and PP
#THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_semrel.py -static -word2vec -img_w200 -l1_nhu100 -pad7 -$cat -n_runs20 -n_train10000000 > ../result/segcnn_${cat}

#echo "___ tep:"
#cat="tep"
#python /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/src/datagen.py ${cat} # because of different settings for TrP, TeP and PP
#THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_semrel.py -static -word2vec -img_w500 -l1_nhu150 -pad4 -$cat -n_runs20 -n_train10000000 > ../result/segcnn_${cat}

#echo "___ pp:"
#cat="pp"
#python /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/src/datagen.py ${cat} # because of different settings for TrP, TeP and PP
#THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_semrel.py -static -word2vec -img_w400 -l1_nhu100 -pad10 -$cat -n_runs20 -n_train10000000 > ../result/segcnn_${cat}




#set scale_fac in datagen.py to 100; multiply scale_fac in semclass() call in cnn_preprocess.py by 0; check that compa() call does not use multiplication with 0
echo "___ trp:"
cat="trp"
python /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/src/datagen.py ${cat} # because of different settings for TrP, TeP and PP
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_semrel.py -static -word2vec -img_w200 -l1_nhu100 -pad7 -$cat -n_runs20 -n_train10000000 > ../result/compa_${cat}

echo "___ tep:"
cat="tep"
python /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/src/datagen.py ${cat} # because of different settings for TrP, TeP and PP
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_semrel.py -static -word2vec -img_w500 -l1_nhu150 -pad4 -$cat -n_runs20 -n_train10000000 > ../result/compa_${cat}

echo "___ pp:"
cat="pp"
python /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/seg_cnn/src/datagen.py ${cat} # because of different settings for TrP, TeP and PP
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_semrel.py -static -word2vec -img_w400 -l1_nhu100 -pad10 -$cat -n_runs20 -n_train10000000 > ../result/compa_${cat}
