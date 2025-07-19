ini untuk raspberry pi 4. Link tutorialnya: https://www.youtube.com/watch?v=lJRLzXf7EVs
ini di ambil dari https://github.com/freedomwebtech/yolo11-edgetpu/tree/main   dan   https://docs.ultralytics.com/guides/coral-edge-tpu-on-raspberry-pi/#can-i-export-my-ultralytics-yolo11-model-to-be-compatible-with-coral-edge-tpu

sedangkan untuk menjalankannya di ubuntu 22.04, maka lakukan perintah berikut ini
1. buat envi pyhton 3.8.19
2. pip install ultralytics ncnn  [lama kali installnya karena 2GB]
3. pip install opencv-contrib-python
4. pip install cvzone
5. lalu ikuti acuan dari https://github.com/google-coral/edgetpu/issues/771 <br>
   pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp38-cp38-linux_x86_64.whl <br>
   pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp38-cp38-linux_x86_64.whl <br>
selesai
