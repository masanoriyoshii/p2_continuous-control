

## Step 01


```
scores = ddpg(n_episodes=50, max_t=1000, print_every=1)
```

```
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
```

```
Episode 1	Average Score: 0.34
Episode 2	Average Score: 0.54
Episode 3	Average Score: 0.77
Episode 4	Average Score: 1.17
Episode 5	Average Score: 1.03
Episode 6	Average Score: 1.22
Episode 7	Average Score: 0.77
Episode 8	Average Score: 1.21
Episode 9	Average Score: 0.80
Episode 10	Average Score: 0.70
Episode 11	Average Score: 0.82
Episode 12	Average Score: 0.78
Episode 13	Average Score: 0.99
Episode 14	Average Score: 1.55
Episode 15	Average Score: 1.32
Episode 16	Average Score: 1.32
Episode 17	Average Score: 1.46
Episode 18	Average Score: 1.13
Episode 19	Average Score: 1.11
Episode 20	Average Score: 0.84
Episode 21	Average Score: 1.24
Episode 22	Average Score: 0.83
Episode 23	Average Score: 0.95
Episode 24	Average Score: 0.77
Episode 25	Average Score: 0.66
Episode 26	Average Score: 0.61
Episode 27	Average Score: 0.44
Episode 28	Average Score: 0.61
Episode 29	Average Score: 0.56
Episode 30	Average Score: 0.72
Episode 31	Average Score: 0.69
Episode 32	Average Score: 0.61
Episode 33	Average Score: 0.72
Episode 34	Average Score: 0.58
Episode 35	Average Score: 0.42
Episode 36	Average Score: 0.44
Episode 37	Average Score: 0.57
Episode 38	Average Score: 0.56
Episode 39	Average Score: 0.54
Episode 40	Average Score: 0.78
Episode 41	Average Score: 0.62
Episode 42	Average Score: 0.67
Episode 43	Average Score: 0.49
Episode 44	Average Score: 0.54
Episode 45	Average Score: 0.65
Episode 46	Average Score: 0.59
Episode 47	Average Score: 0.45
Episode 48	Average Score: 0.66
Episode 49	Average Score: 0.65
Episode 50	Average Score: 0.58
```


![p2_result_00](https://user-images.githubusercontent.com/4464676/79061937-563b7380-7cd0-11ea-8c5b-f1c7a8f72755.png)


## Step 02

- batch nomalization
- tweak learning rate

```
scores = ddpg(n_episodes=50, max_t=1000, print_every=1)
```

```
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
```

```
Episode 1	Average Score: 0.82
Episode 2	Average Score: 1.58
Episode 3	Average Score: 1.99
Episode 4	Average Score: 1.96
Episode 5	Average Score: 2.55
Episode 6	Average Score: 2.29
Episode 7	Average Score: 3.98
Episode 8	Average Score: 4.33
Episode 9	Average Score: 3.67
Episode 10	Average Score: 4.56
Episode 11	Average Score: 5.42
Episode 12	Average Score: 5.92
Episode 13	Average Score: 3.97
Episode 14	Average Score: 5.01
Episode 15	Average Score: 4.02
Episode 16	Average Score: 4.77
Episode 17	Average Score: 5.13
Episode 18	Average Score: 5.88
Episode 19	Average Score: 6.00
Episode 20	Average Score: 6.19
Episode 21	Average Score: 6.70
Episode 22	Average Score: 6.11
Episode 23	Average Score: 6.70
Episode 24	Average Score: 6.69
Episode 25	Average Score: 7.30
Episode 26	Average Score: 7.06
Episode 27	Average Score: 7.65
Episode 28	Average Score: 6.86
Episode 29	Average Score: 6.54
Episode 30	Average Score: 6.35
Episode 31	Average Score: 7.61
Episode 32	Average Score: 7.66
Episode 33	Average Score: 7.95
Episode 34	Average Score: 8.01
Episode 35	Average Score: 8.48
Episode 36	Average Score: 8.86
Episode 37	Average Score: 8.67
Episode 38	Average Score: 8.22
Episode 39	Average Score: 8.40
Episode 40	Average Score: 8.30
Episode 41	Average Score: 9.32
Episode 42	Average Score: 8.15
Episode 43	Average Score: 9.47
Episode 44	Average Score: 8.46
Episode 45	Average Score: 8.74
Episode 46	Average Score: 9.55
Episode 47	Average Score: 8.60
Episode 48	Average Score: 10.00
Episode 49	Average Score: 8.38
Episode 50	Average Score: 10.89
```

![p2_result_01](https://user-images.githubusercontent.com/4464676/79072432-f8cd1400-7d1b-11ea-87b0-cca7546cab1e.png)


## Step 03

- update every 20 timesteps
- update 10

```
scores = ddpg(n_episodes=50, max_t=1000, print_every=1)
```

```
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 20
TIMES_LEARN = 10
```

```
Episode 1	Average Score: 0.50
Episode 2	Average Score: 1.15
Episode 3	Average Score: 2.59
Episode 4	Average Score: 2.49
Episode 5	Average Score: 2.79
Episode 6	Average Score: 3.54
Episode 7	Average Score: 2.80
Episode 8	Average Score: 2.57
Episode 9	Average Score: 3.14
Episode 10	Average Score: 3.57
Episode 11	Average Score: 5.28
Episode 12	Average Score: 5.01
Episode 13	Average Score: 4.83
Episode 14	Average Score: 5.45
Episode 15	Average Score: 5.60
Episode 16	Average Score: 6.05
Episode 17	Average Score: 7.09
Episode 18	Average Score: 6.42
Episode 19	Average Score: 5.63
Episode 20	Average Score: 5.99
Episode 21	Average Score: 6.28
Episode 22	Average Score: 5.83
Episode 23	Average Score: 6.34
Episode 24	Average Score: 6.39
Episode 25	Average Score: 5.59
Episode 26	Average Score: 6.94
Episode 27	Average Score: 7.22
Episode 28	Average Score: 7.10
Episode 29	Average Score: 5.39
Episode 30	Average Score: 6.89
Episode 31	Average Score: 8.08
Episode 32	Average Score: 7.78
Episode 33	Average Score: 7.34
Episode 34	Average Score: 8.66
Episode 35	Average Score: 7.24
Episode 36	Average Score: 8.75
Episode 37	Average Score: 8.60
Episode 38	Average Score: 7.25
Episode 39	Average Score: 6.69
Episode 40	Average Score: 7.54
Episode 41	Average Score: 8.38
Episode 42	Average Score: 7.47
Episode 43	Average Score: 8.53
Episode 44	Average Score: 8.90
Episode 45	Average Score: 9.33
Episode 46	Average Score: 7.97
Episode 47	Average Score: 7.15
Episode 48	Average Score: 8.39
Episode 49	Average Score: 7.25
Episode 50	Average Score: 8.41
```

![p2_result_02](https://user-images.githubusercontent.com/4464676/79087474-52642b80-7d7a-11ea-9b54-872db1fc272b.png)

## Step 03
- use gradient clipping when training the critic network

```
scores = ddpg(n_episodes=150, max_t=1000, print_every=5)
```

```
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 20
TIMES_LEARN = 10
```

```
Episode 5	Average Score: 2.79
Episode 10	Average Score: 3.57
Episode 15	Average Score: 5.60
Episode 20	Average Score: 5.99
Episode 25	Average Score: 5.59
Episode 30	Average Score: 6.89
Episode 35	Average Score: 7.24
Episode 40	Average Score: 7.54
Episode 45	Average Score: 9.33
Episode 50	Average Score: 8.41
Episode 55	Average Score: 9.50
Episode 60	Average Score: 11.78
Episode 65	Average Score: 10.17
Episode 70	Average Score: 11.05
Episode 75	Average Score: 12.11
Episode 80	Average Score: 10.43
Episode 85	Average Score: 12.37
Episode 90	Average Score: 13.62
Episode 95	Average Score: 15.08
Episode 100	Average Score: 14.80
Episode 105	Average Score: 13.33
Episode 110	Average Score: 14.77
Episode 115	Average Score: 18.76
Episode 120	Average Score: 19.17
Episode 125	Average Score: 21.75
Episode 130	Average Score: 22.14
Episode 135	Average Score: 24.26
Episode 140	Average Score: 25.81
Episode 145	Average Score: 23.38
Episode 150	Average Score: 24.65
```

![p2_result_03](https://user-images.githubusercontent.com/4464676/79123033-2aa1b180-7dd4-11ea-916f-ad692a6f90b4.png)
