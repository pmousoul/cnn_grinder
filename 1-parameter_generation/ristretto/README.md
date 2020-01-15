## Ristretto SqueezeNet v1.1 model and example files

Assuming that you have correctly built and configured Caffe-Ristretto and the ILSVRC 2012 dataset, this part of the project contains the required files and scripts to quantize and benchmark the floating and fixed-point models.

Specifically, in the `models/SqueezeNet_v1.1/` directory are:

* the `deploy.prototxt`, the `train_val.prototxt`, and the `squeezenet_v1.1.caffemodel` downloaded from: https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1 (use the `download_model_files.sh` script) - make sure to properly modify the .prototxt files to read the ILSVRC 2012 dataset installed and configured in your PC.


In the `models/SqueezeNet_v1.1/RistrettoDemo/` directory are:

* the `solver_finetune.prototxt` file which it should be useful for fine-tuning

* the `quantized.prototxt` file which is generated after executing `./examples/ristretto/00_quantize_squeezenet_v1.1.sh` from the <caffe-ristretto_root>; example last lines of the output of this script can be found in `quantization_network_accuracy_analysis_results.txt`

* the `quantized_(d)fixed.prototxt` file is a modified version of the `quantized.prototxt` file; it describes the ristretto quantization layers according to the (dynamic) fixed-point parameter choices; this file along with the `squeezenet_v1.1.caffemodel` file can be used by the `02_benchmark_(d)fixedpoint_squeezenet_v1.1.sh` to benchmark the network accuracy of the quantized network description


In the `examples/ristretto/` directory can be found similar scripts that their function is already described above.

For more information related to the Ristretto tool, see: http://lepsucd.com/?page_id=621