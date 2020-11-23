Requirements
--

1. Python 3.3-3.7
2. CUDA 10.0
3. CuDNN 7.4
4. All required packages installed using the command ```pip install requirements.txt```


GAN prediction
--

To run the pretrained network, use one of the following commands. The results will be saved to the `results` directory.

1. LPIPS Ours, far periphery:  
```
python run_gan.py predict -d results -c pretrained/lpips_dist_far -t dataset/test_far
```
2. LPIPS Ours, near periphery: 
```
python run_gan.py predict -d results -c pretrained/lpips_dist_near -t dataset/test_near
```
3. LPIPS, Far periphery:   
```
python run_gan.py predict -d results -c pretrained/lpips_real_far -t dataset/test_far
```
4. LPIPS, near periphery:  
```
python run_gan.py predict -d results -c pretrained/lpips_real_near -t dataset/test_near
```
5. L2 Ours, far periphery:     
```
python run_gan.py predict -d results -c pretrained/l2_dist_far -t dataset/test_far
```
6. L2 Ours, near periphery:    
```
python run_gan.py predict -d results -c pretrained/l2_dist_near -t dataset/test_near
```
7. L2, far periphery:     
```
python run_gan.py predict -d results -c pretrained/l2_real_far -t dataset/test_far
```
8. L2, near periphery:    
```
python run_gan.py predict -d results -c pretrained/l2_real_near -t dataset/test_near
```

GAN Training
--

Use the following command to run the network training: 
```
run_gan.py [-h] [-p [lpips, l2]] [-g GROUND_TRUTH]
                              [-i INPUT] [-r REAL] [-d RESULTS]
                              [-c CHECKPOINTS] [-l LOG] [-e EPOCHS]
                              [-t TEST_INPUT] [-b BATCH_SIZE]
                              [predict, train]
```

Positional arguments:
  1. `[predict, train]` specify type of action to perform on a network

Optional arguments:
  1. `-p [lpips, l2], --perceptual_loss_type [lpips, l2]` used loss metric for the generator
  2. `-g GROUND_TRUTH, --ground_truth GROUND_TRUTH` the images folder used as a ground truth for the generator
  3.  `-i INPUT, --input INPUT` the images folder used as an input for the generator
  4. `-r REAL, --real REAL`  the images folder treated as 'real' by the discriminator
  5. `-d RESULTS, --directory RESULTS` the folder in which all generated test images would be saved
  6. `-c CHECKPOINTS, --checkpoints CHECKPOINTS` the folder from which to load the checkpoints and where to save the new ones
  7. `-l LOG, --log LOG` the folder to which the tensorboard log will be saved
  8. `-e EPOCHS, --epochs EPOCHS` specify number of epochs for training
  9. `-t TEST_INPUT, --test_input TEST_INPUT` the images folder used for testing the network each epoch
  10. `-b BATCH_SIZE, --batch_size BATCH_SIZE` number of images put in a batch during training

Images used as a training dataset must have a size of 256x256.

To train the network, custom dataset is required. It can be generated using the  Interpolation and Image synthesis script. Additionally, we provide the LPIPS pretrained network from https://github.com/richzhang/PerceptualSimilarity in a zip file in the GAN/ directory, which needs to be extracted inplace. Examples of available training procedures:

1. LPIPS Ours, far periphery:  
```
python run_gan.py train -p lpips -g dataset/ground_truth -i dataset/train_far -r dataset/distorted_far -d results/train_lpips_distorted_far -c checkpoints/train_lpips_distorted_far -l logs/train_lpips_distorted_far -t dataset/test_far -b 8 -e 20
```
2. LPIPS Ours, near periphery: 
```
python run_gan.py train -p lpips -g dataset/ground_truth -i dataset/train_near -r dataset/distorted_near -d results/train_lpips_distorted_near -c checkpoints/train_lpips_distorted_near -l logs/train_lpips_distorted_near -t dataset/test_near -b 8 -e 20
```
3. LPIPS, far periphery:   
```
python run_gan.py train -p lpips -g dataset/ground_truth -i dataset/train_far -r dataset/ground_truth -d results/train_lpips_standard_far -c checkpoints/train_lpips_standard_far -l logs/train_lpips_standard_far -t dataset/test_far -b 8 -e 20
```
4. LPIPS, near periphery:  
```
python run_gan.py train -p lpips -g dataset/ground_truth -i dataset/train_near -r dataset/ground_truth -d results/train_lpips_standard_near -c checkpoints/train_lpips_standard_near -l logs/train_lpips_standard_near -t dataset/test_near -b 8 -e 20
```
5. L2 Ours, far periphery:    
```
python run_gan.py train -p l2 -g dataset/ground_truth -i dataset/train_far -r dataset/distorted_far -d results/train_l2_distorted_far -c checkpoints/train_lpips_distorted_far -l logs/train_lpips_distorted_far -t dataset/test_far -b 8 -e 20
```
6. L2 Ours, near periphery:   
```
python run_gan.py train -p l2 -g dataset/ground_truth -i dataset/train_near -r dataset/distorted_near -d results/train_l2_distorted_near -c checkpoints/train_lpips_distorted_near -l logs/train_lpips_distorted_near -t dataset/test_near -b 8 -e 20
```
7. L2, far periphery:     
```
python run_gan.py train -p l2 -g dataset/ground_truth -i dataset/train_far -r dataset/ground_truth -d results/train_l2_standard_far -c checkpoints/train_l2_standard_far -l logs/train_l2_standard_far -t dataset/test_far -b 8 -e 20
```
8. L2, near periphery:    
```
python run_gan.py train -p l2 -g dataset/ground_truth -i dataset/train_near -r dataset/ground_truth -d results/train_l2_standard_near -c checkpoints/train_l2_standard_near -l logs/train_l2_standard_near -t dataset/test_near -b 8 -e 20
```


Interpolation
--

Used for generating subsampled interpolated images used as an input to the GAN network. Usage: 
```
generate_interpolated_data.py [-h] [-s SAMPLING] [-o OUTPUT] [-i INPUT]
```
Optional arguments:
  1. `-s SAMPLING, --sampling SAMPLING` sampling rate of the synthesis
  2. `-o OUTPUT, --output OUTPUT` the folder for generated images
  3. `-i INPUT, --input INPUT` the folder with input images



Example usage: 
```
python generate_interpolated_data.py -s 0.007 -o dataset/interpolated_0.007 -i dataset/ground_truth
```

Image synthesis
--

Performes image synthesis using a specified number of guided samples. Usage: 
```
texture_synthesis.py [-h] [-s SAMPLING] [-o OUTPUT] [-i INPUT]
```

Optional arguments:
  1. `-s SAMPLING, --sampling SAMPLING` guided sampling rate of the synthesis
  2. `-o OUTPUT, --output OUTPUT` the folder for generated images
  3. `-i INPUT, --input INPUT` the folder with input images

Example usage: 
```
python texture_synthesis.py -s 0.007 -o dataset/synthesized_0.007 -i dataset/ground_truth
```
