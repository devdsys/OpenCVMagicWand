# OpenCVMagicWand
Control processes by drawing shapes with a light source | Python, OpenCV

Experience the sensation of being a wizard with a real magic wand... but using technology instead of magic :)

Control anything you want: music playback, your smart home, or anything else managed by code. All you need is a device (preferably with a display), a camera, and some source of light for drawing. You can execute any piece of code by drawing the relevant figure.

Feel free to use any shapes and as many figures as you want. However, you'll need to find or collect a relevant dataset and train a classification neural network for those figures.

**In my case, I've written scripts to control two lights (left red and right green) using a Raspberry Pi. However, you can modify the code for any device compatible with Python and OpenCV, and write any other behavior for your class predictions.**

For my project I first collected a dataset of my custom figures using [written script](https://github.com/devdsys/OpenCVMagicWand/blob/main/magicWandCollection.py).

For data collection you can drow a figures on your screen using some source of light. In my case i've been using an LED that is turned on by a button and powered by a power bank.

![Images capturing process](https://github.com/devdsys/OpenCVMagicWand/blob/main/README_content/data_collection.gif)   

And this is what the saved image looks like:

<img src="https://github.com/devdsys/OpenCVMagicWand/blob/main/README_content/saved_image_example.jpg" width="300" height="250">  

**Drawing process:**

On every frame, image processing is done to find a spot of light. If detected, the center of the spot is determined and stored in a list. As long as the spot remains visible in each subsequent frame, the process continues to save the spot's center point into the list and draw a sequential line between each pair of points.

If the light spot disappears in the next frame, draw the same lines on a blank image and save the result.

Repeat the process.

**Training process:**

Based on the collected dataset of images, I've trained a classification neural network. Since the whole project was centered around the concept of drawing using light and OpenCV, minimal time was dedicated to network training. This is because for each new figure or your custom set of figures, you would need to retrain the model. Because of very small set of images, I've used a pre-trained model for handwritten digit classification based on CIFAR-10 and used transfer learning with fine-tuning to retrain the model for my own dataset. Since I wasn't primarily concerned with the model's high accuracy, I saved the first-try trained model and continued working with it. Finally, I converted the trained model to TensorFlowLite for use with a Raspberry Pi.

If you wish to replicate this project, please search for a tutorial on how to train a multiclass classification neural network or feel free to contact me.

**Inference:**

![Controll two lights using magic wand](https://github.com/devdsys/OpenCVMagicWand/blob/main/README_content/inference.gif)   

For inference, I used [script](https://github.com/devdsys/OpenCVMagicWand/blob/main/magicWandInference.py) similar to the collection process, but instead of saving the drawn image, I used it to make predictions. Additionally, I modified the code to control Raspberry Pi pins (to turn on and off lights using relay).


**Prons of project: The project can be used for own custom shapes, models, and can be adjusted for different devices. Also, you can find alredy trained model on some shapes and control your processes using shapes, on which the model was trained.**

**Cons: You need to train own model for own set on shapes. Scripts are very sensitive to any sources or spots of light.** 
