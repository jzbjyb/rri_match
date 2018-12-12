#TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#g++ -std=c++11 -shared jumper.cc -o jumper.so -fPIC -I $TF_INC -O2
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

i1=jumper.cc
o1=${i1/.cc/.so}
i2=dynamic_split.cc
o2=${i2/.cc/.so}

g++ -std=c++11 -shared $i1 -o $o1  -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2  -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 -shared $i2 -o $o2  -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2  -D_GLIBCXX_USE_CXX11_ABI=0
