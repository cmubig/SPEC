# Social Pattern Extraction Convolution (SPEC) 

This software implements the Social Pattern Extraction Convolution (SPEC) algorithm published in the following paper:<br>

<p>
<a href="https://scholar.google.com/citations?user=opaOHYwAAAAJ">Dapeng Zhao</a> and
<a href="http://www.cs.cmu.edu/~jeanoh">Jean Oh</a>,
<a href="https://arxiv.org/abs/1803.10892"><b>Noticing Motion Patterns: A Temporal CNN With a Novel Convolution Operator for Human Trajectory Prediction</b></a>
<a href="https://ieeexplore.ieee.org/abstract/document/9309403">IEEE Robotics and Automation Letters (RA-L)</a>, December 2020.
</p>

<p>
  <pre>
@article{zhao2020noticing,
  title={Noticing Motion Patterns: Temporal CNN with a Novel Convolution Operator for Human Trajectory Prediction},
  author={Zhao, Dapeng and Oh, Jean},
  journal={IEEE Robotics and Automation Letters},
  year={2020},
  publisher={IEEE}
}
</pre>
</p>

**Social Pattern Extraction Convolution (SPEC)<sub>[1]</sub> is an algorithm designed and implemented to help robots to predict humans’ future trajectories, so that robots can navigate in a safe and self-explanatory manner, while humans feel comfortable about robots’ motion.**  

System Functionality:  

- Can be trained on the ETH pedestrian dataset<sub>[2]</sub> and UCY pedestrian dataset<sub>[3]</sub>, which are provided together with the software in the “./data” folder    
- Can be trained on other datasets, as long as given in the same format as the provided ETH/UCY datasets  
- Can predict future trajectories of human pedestrians, when given observed history trajectories  
- Can be used to simulate human behaviors, when given appropriate initialization, e.g. a few steps of locations, which is useful for social navigation study  
- Can evaluate given trajectories, i.e. report their similarities to the trajectories used for training, which can potentially be useful to surveillance and anomaly detection study  



## Project Contributors  
Dapeng Zhao

PI: [Jean Oh](http://www.cs.cmu.edu/~jeanoh/)

---

## Download and install  
```
git clone https://github.com/cmubig/SPEC.git
cd SPEC
```  
## Library Requirement and Suggested Version

* numpy==1.18.2+
* torch==1.4.0

## Hardware, Software System Requirements  
These are the specifications of the hardware used during development.  
Different combinations of hardware could work and will yield different speeds.  

* Intel(R) Core(TM) i9-9820X CPU @ 3.30GHz
* 16GB ram x 8
* GeForce RTX 2080 Ti 

These are the specifications of the software used during development.
Different combinations of software might work.  

* Ubuntu 18.04
* CUDA 11.0 

---
## Training Performance
Below are some  empirical observation based non-functional expectations which users can have when trying to utilize the provided software:  

* Default training procedure (Dataset: zara1, number of epochs: 100, batch size: 64) could take ~180 seconds  
* Inference time of one trajectory is ~5 milliseconds  

## Usage
To train, run:  
```python train.py```  

To train with different setting, add a token in Arguments.py with your desired setting, and run, e.g. token as "hotel":
```python train.py --token "hotel"```

Functions for inference can be found in model.py.  

---

## References

[1] D. Zhao and J. Oh, "Noticing Motion Patterns: A Temporal CNN With a Novel Convolution Operator for Human Trajectory Prediction," in IEEE Robotics and Automation Letters, vol. 6, no. 2, pp. 628-634, April 2021, doi: 10.1109/LRA.2020.3047771.  
https://ieeexplore.ieee.org/abstract/document/9309403/  

[2] S. Pellegrini, A. Ess, K. Schindler and L. van Gool, "You'll never walk alone: Modeling social behavior for multi-target tracking," 2009 IEEE 12th International Conference on Computer Vision, Kyoto, Japan, 2009, pp. 261-268, doi: 10.1109/ICCV.2009.5459260.  
https://ieeexplore.ieee.org/abstract/document/5459260  

[3] A. Lerner, Y. Chrysanthou, and D. Lischinski, “Crowds by example,” in Computer graphics forum, vol. 26, no. 3. Wiley Online Library, 2007, pp. 655–664.  
https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-8659.2007.01089.x  

