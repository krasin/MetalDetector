# MetalDetector
GoogLeNet on iOS demo

This is a basic demo of GoogLeNet from Caffe Zoo working on iOS.
It takes the input from the camera and prints the top prediction.
The speed is 2.5 seconds per frame on iPhone 6S, and 7 seconds on iPhone 6S.

Two known bugs:

* You need to hold the phone so that the round button is on the right.
Otherwise, the network will get a rotated image, and the classification will likely miss.

* There's a rounding bug right now (likely, in Convolution layers). While the network gives answers
which are within ~8% of Caffe output, it's noticeably worse than a real GoogLeNet taken from the Caffe Zoo.

Also, please, be aware that ImageNet classes are weird, not so many real world things could be detected (but it knows about 300 breeds of dogs, whee!)
