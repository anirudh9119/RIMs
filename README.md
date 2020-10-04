 # Recurrent Independent Mechanisms
 An implementation of [Recurrent Independent Mechanisms](https://arxiv.org/abs/1909.10893) (Goyal et al. 2019) in PyTorch.
 
[Anirudh Goyal](https://anirudh9119.github.io/), [Alex Lamb](https://alexlamb62.github.io/), [Jordan Hoffmann](https://jhoffmann.org/), [Shagun Sodhani](https://mila.quebec/en/person/shagun-sodhani/), [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/), [Yoshua Bengio](https://mila.quebec/en/yoshua-bengio/), [Bernhard Sch{\"o}lkopf](https://www.is.mpg.de/~bs)
 
 It features adding and copying synthetic task from the paper. It also features the code to reproduce atari results.
 
 
 # Examples
 `./experiment_copying.sh 600 6 4 50 200` for full training & test run of RIMs on the copying task.
 
 `./experiment_adding.sh 600 6 4 50 200 0.2` for full training and test run of RIMs on the adding task. 




    @article{goyal2019recurrent,
        title={Recurrent independent mechanisms},
        author={Goyal, Anirudh and Lamb, Alex and Hoffmann, Jordan and Sodhani, Shagun and Levine, Sergey and Bengio, Yoshua and Sch{\"o}lkopf, Bernhard},
        journal={arXiv preprint arXiv:1909.10893},
        year={2019}
    }
