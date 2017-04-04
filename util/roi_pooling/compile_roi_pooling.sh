TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

nvcc -std=c++11 -c -o roi_pooling_op_gpu.cu.o roi_pooling_op_gpu.cu.cc \
-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++-4.8 -std=c++11 -shared -o ./build/roi_pooling.so roi_pooling_op.cc \
roi_pooling_op_gpu.cu.o -I $TF_INC -fPIC -lcudart -L/usr/local/cuda-8.0/lib64

rm -rf roi_pooling_op_gpu.cu.o
