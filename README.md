ini untuk raspberry pi 4. Link tutorialnya: https://www.youtube.com/watch?v=lJRLzXf7EVs
ini di ambil dari https://github.com/freedomwebtech/yolo11-edgetpu/tree/main

sedangkan untuk menjalankannya di ubuntu 22.04, maka lakukan perintah berikut ini
1. buat envi pyhton 3.8.19
2. pip install ultralytics ncnn  [lama kali installnya karena 2GB]
3. pip install opencv-contrib-python
4. lalu ikuti acuan dari https://github.com/google-coral/edgetpu/issues/771
   pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp38-cp38-linux_x86_64.whl
   pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp38-cp38-linux_x86_64.whl
selesai
