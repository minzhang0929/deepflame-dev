#!/bin/sh
if [ -e libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip ]
then
    echo "libtorch.zip exist."
else
    wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcpu.zip
fi
if [ -d "thirdParty/libtorch" ]; then
    echo "libtorch exist."
else
    unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip -d thirdParty
fi


cp bashrc.raw bashrc
sed -i s#pwd#$PWD#g ./bashrc
sed -i s#CONDA_PREFIX#$CONDA_PREFIX#g ./bashrc
TORCH_DIR=$PWD/thirdParty/libtorch
sed -i s#TORCH_DIR#$TORCH_DIR#g ./bashrc



if [ -d "src_orig" ]; then
    echo "src_orig exist."
else
    mkdir -p src_orig/MomentumTransportModels
    mkdir -p src_orig/thermophysicalModels
    mkdir -p src_orig/thermophysicalModels/specie
    mkdir -p src_orig/fvModels
    mkdir -p src_orig/lagrangian
    #mkdir -p src_orig/regionModels
    cp -r $FOAM_SRC/MomentumTransportModels/compressible src_orig/MomentumTransportModels
    cp -r $FOAM_SRC/thermophysicalModels/basic src_orig/thermophysicalModels
    cp -r $FOAM_SRC/thermophysicalModels/thermophysicalProperties src_orig/thermophysicalModels
    cp -r $FOAM_SRC/thermophysicalModels/specie/thermophysicalFunctions/ src_orig/thermophysicalModels/specie
    cp -r $FOAM_SRC/fvModels src_orig
    cp -r $FOAM_SRC/lagrangian/parcel src_orig/lagrangian
    cp -r $FOAM_SRC/lagrangian/parcelTurbulence src_orig/lagrangian
    #cp -r $FOAM_SRC/regionModels/surfaceFilmModels src_orig/regionModels
fi


source ./bashrc
./Allwmake -j