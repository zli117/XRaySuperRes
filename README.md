# CS446-Project

## Name and NetID:
Shengkun Cui (scui8), Zonglin Li (zli117)


## Inspirations:
* We initially tried a baseline implementation of ESPCN. It achieved a satisfactory result of 9500
* Later we decided to add a denoising network DNCNN. We tried to add it before ESPCN, after ESPCN and both before and after ESPCN. With some tuning of learning rate, we achieved an improvement of ~1000 on public leader board.
* We also tried SRResNet and SRGAN. SRResNet gave us some improvement of several hundred on LB, and SRGAN had problem to converge. 
* When reading some paper and from out experiments, we noticed that L2 loss will cause image to blur. We tried L1 loss on SRResNet, but did not get a significant improvement. 
* Lastly, we tried a new network EDSR. Paired with L1 loss, it achieved a significant improvement, and it's used for final submission.

## Architectures:
* DNCNN + ESPCN:
  * DNCNN -> ESPCN -> DNCNN
  * Run: 
    ```bash
    python dncnn_fine_tune.py -t 16 -b 16 -e 200 -u 200 -y 200 -p <dir for saving checkpoints> -g -i <dir for lr images> -l <dir for hr images> -w <dir for test split> -o <dir for generating processed test split>
    ```
* SRResNet:
  * We tried SRResNet and SRGAN. The training for SRGAN was extremely noisy, and the result is suboptimal. SRResNet, however, yielded pretty good result after 100 epochs.
  * The pipeline has several stages: DNCNN (optional) -> SRResNet -> SRGAN (optional)
  * Run (train with SRGAN, without DNCNN): 
    ```bash
    python srgan.py -t 16 -b 16 -k -c 50 -u 100 -p <dir for saving checkpoints> -g -i <dir for lr images> -l <dir for hr images> -w <dir for test split> -o <dir for generating processed test split>
    ```
  * Run (only with SRResNet): 
    ```bash
    python srgan.py -t 16 -b 16 -k -c 100 -u 0 -p <dir for saving checkpoints> -g -i <dir for lr images> -l <dir for hr images> -w <dir for test split> -o <dir for generating processed test split>
    ```
* EDSR:
  * EDSR gives the best result so far, and we generated our final submission using it.
  * We have tried both L1 loss and L2 loss. And L2 loss produced more visually blurry images.
  * Run the large model with L1 loss: 32 ResBlock, each layer generates 128 feature maps:
    ```bash
    python edsr.py -t 16 -b 16 -c 150 -g -y l1 -p <dir for saving checkpoints> -i <dir for lr images> -l <dir for hr images> -w <dir for test split> -o <dir for generating processed test split>
    ```
  * Run the small model with L1 loss: 16 ResBlock, each layer generates 64 feature maps:
    ```bash
    python edsr.py -t 16 -b 16 -c 150 -g -z -y l1 -p <dir for saving checkpoints> -i <dir for lr images> -l <dir for hr images> -w <dir for test split> -o <dir for generating processed test split>
    ```
    
## Submission log:
`out/history.csv`
