# CS446-Project

## Architectures:

* DNCNN + ESPCN:
  * DNCNN -> ESPCN -> DNCNN
  * Run: 
    `python dncnn_fine_tune.py -t 16 -b 16 -e 200 -u 200 -y 200 -p <dir for saving checkpoints> -g -i <dir for lr images> -l <dir for hr images> -w <dir for test split> -o <dir for generating processed test split>`
* SRResNet:
  * We tried SRResNet and SRGAN. The training for SRGAN was extremely noisy, and the result is suboptimal. SRResNet, however, yielded pretty good result after 100 epochs.
  * The pipeline has several stages: DNCNN (optional) -> SRResNet -> SRGAN (optional)
  * Run (train with SRGAN, without DNCNN): 
    `python srgan -t 16 -b 16 -k -c 50 -u 100 -p <dir for saving checkpoints> -g -i <dir for lr images> -l <dir for hr images> -w <dir for test split> -o <dir for generating processed test split>`
  * Run (only with SRResNet): 
    `python srgan -t 16 -b 16 -k -c 100 -u 0 -p <dir for saving checkpoints> -g -i <dir for lr images> -l <dir for hr images> -w <dir for test split> -o <dir for generating processed test split>`
* EDSR:
  * EDSR gives the best result so far, and we generated our final submission using it.
  * We have tried both L1 loss and L2 loss. And L2 loss produced more visually blurry images.
  * Run the large model with L1 loss: 32 ResBlock, each layer generates 128 feature maps:
    `python edsr.py -t 16 -b 16 -c 150 -g -z -y l1 -p <dir for saving checkpoints> -i <dir for lr images> -l <dir for hr images> -w <dir for test split> -o <dir for generating processed test split>`
  * Run the small model with L1 loss: 16 ResBlock, each layer generates 64 feature maps:
    `python edsr.py -t 16 -b 16 -c 150 -g -y l1 -p <dir for saving checkpoints> -i <dir for lr images> -l <dir for hr images> -w <dir for test split> -o <dir for generating processed test split>`

