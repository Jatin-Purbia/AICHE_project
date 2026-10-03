# Stage 1 results summary

All numbers below were produced by `scripts/run_all.py` (mode: full).

## Software and hardware
- Python 3.12.4, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, torch 2.14.1+cpu, pandas 3.0.6
- CPU: Intel(R) Core(TM) 7 150U (2 torch threads), Windows-11-10.0.26200-SP0

## Datasets
- Steady state: 2000 LHS points, 2000 retained (dropped: none); split 1400/300/300.
- Dynamic: 30 trajectories x 2000 s; split by trajectory 20/5/5.

## Steady-state models (test set)

- MLR: RMSE [K,K,K,V] = 1.731, 2.191, 2.375, 0.03795; mean R2 = 0.8482; max |e_T| = 11.70 K
- Poly-ridge: RMSE [K,K,K,V] = 0.7670, 0.9644, 1.040, 0.01432; mean R2 = 0.9728; max |e_T| = 5.406 K
- MLP: RMSE [K,K,K,V] = 0.04206, 0.03577, 0.04269, 0.0004443; mean R2 = 1.000; max |e_T| = 0.2790 K
- TSK / ANFIS-type: RMSE [K,K,K,V] = 0.3065, 0.3866, 0.4170, 0.007674; mean R2 = 0.9949; max |e_T| = 2.030 K
- **Best steady-state model (lowest mean temperature RMSE): MLP.** Its NRMSE per output = 0.001311, 0.001296, 0.001606, 0.001083.

## Dynamic models (test trajectories)

One-step (teacher forced):

- ARX: RMSE [K] = 0.00008086, 0.00004889, 0.0004265; R2_delta = 0.9956
- NARX-MLP: RMSE [K] = 0.00002409, 0.00002169, 0.0001522; R2_delta = 0.9994
- LSTM: RMSE [K] = 0.00006978, 0.0001000, 0.0005875; R2_delta = 0.9895
- PINC: RMSE [K] = 0.00001853, 0.00003359, 0.0002528; R2_delta = 0.9984

Free-run rollout, 2000 s (RMSE pooled over the 3 temperatures at horizon H; per-state RMSE over the full horizon):

- ARX: H=10: 0.03793 K, H=50: 0.08044 K, H=100: 0.1039 K, H=500: 0.2555 K, H=2000: 0.4585 K; full per state [K] = 0.2251, 0.3771, 0.6615
- NARX-MLP: H=10: 0.0031 K, H=50: 0.009885 K, H=100: 0.01170 K, H=500: 0.009358 K, H=2000: 0.01174 K; full per state [K] = 0.01062, 0.008339, 0.01493
- LSTM: H=10: 0.01271 K, H=50: 0.04313 K, H=100: 0.0565 K, H=500: 0.0416 K, H=2000: 0.0383 K; full per state [K] = 0.02705, 0.03777, 0.04588
- PINC: H=10: 0.0005245 K, H=50: 0.002207 K, H=100: 0.003631 K, H=500: 0.01263 K, H=2000: 0.02253 K; full per state [K] = 0.009570, 0.02253, 0.03036
- **Best dynamic model by full-horizon rollout RMSE: NARX-MLP.**

### Rollout drift behaviour
Mean instantaneous pooled RMSE at t = 100, 500, 2000 s and log-log growth exponent p between 100 and 2000 s (error ~ t^p; p near 0 = bounded offset, p near 1 = linear drift, p > 1 = accelerating):

- ARX: 0.1427, 0.4066, 0.5721 K; p = 0.46 (roughly linear drift)
- NARX-MLP: 0.01197, 0.00992, 0.01621 K; p = 0.10 (bounded / saturating)
- LSTM: 0.06242, 0.03159, 0.03988 K; p = -0.15 (bounded / saturating)
- PINC: 0.00568, 0.02445, 0.02694 K; p = 0.52 (roughly linear drift)

## PINC
- Training time per seed: 552.1 s (mean over seeds; all seeds: [432, 734, 490]).
- Final training physics loss (normalised residual^2, last L-BFGS evaluation): ['2.271e-05', '2.233e-05', '1.992e-05'].
- Mean squared normalised ODE residual of the learned map on test states: ['9.989e-06', '1.062e-05', '9.976e-06'] (RMS residual ['1.533e-04', '1.683e-04', '1.834e-04'] K/s).
- IC mode: hard (hard = initial condition satisfied exactly).

## Cost (inference per 1000 rollout steps, median of repeats, batch of 1)
- RK4 simulator: 0.09260 s
- ARX: 0.04543 s (0.49 x RK4), parameters 39, training time 0.05249 s
- NARX-MLP: 0.07533 s (0.81 x RK4), parameters 5187, training time 13.62 s
- LSTM: 0.3607 s (3.89 x RK4), parameters 5219, training time 60.20 s
- PINC: 0.1286 s (1.39 x RK4), parameters 13187, training time 552.1 s

## Noise sensitivity (sigma_n = 0.05 K added to measured test states, models trained on clean data)
Rollout RMSE [K] from the noisy measured initial state vs the clean truth (noise enters only through the initial state, plus the initial lag/window for the lagged models):

- ARX: clean 0.4585, noisy 0.4612 (1.01 x). One-step teacher-forced on noisy measured states (target = clean x_k+1): clean 0.0002522 K, noisy 0.1088 K
- NARX-MLP: clean 0.01174, noisy 0.02969 (2.53 x). One-step teacher-forced on noisy measured states (target = clean x_k+1): clean 0.00008990 K, noisy 0.04944 K
- LSTM: clean 0.0383, noisy 0.04917 (1.28 x). One-step teacher-forced on noisy measured states (target = clean x_k+1): clean 0.0003465 K, noisy 0.04952 K
- PINC: clean 0.02253, noisy 0.03226 (1.43 x). One-step teacher-forced on noisy measured states (target = clean x_k+1): clean 0.0001478 K, noisy 0.04936 K

## Step responses (RMSE vs simulator over 500 s, seed 0, per state [K])

- step $I$: 30 $\to$ 45 A: ARX: 0.09398/0.00648/0.02212; NARX-MLP: 0.002098/0.004196/0.003264; LSTM: 0.007896/0.01730/0.01950; PINC: 0.01026/0.002424/0.005619
- step $u_1$: 0.6 $\to$ 0.9: ARX: 0.05355/0.01909/0.08614; NARX-MLP: 0.0005433/0.0003323/0.01183; LSTM: 0.0002934/0.01133/0.04714; PINC: 0.002304/0.004021/0.004035
- step $u_2$: 0.6 $\to$ 0.9: ARX: 0.06882/0.07009/0.3506; NARX-MLP: 0.002349/0.0005665/0.002523; LSTM: 0.01359/0.01731/0.01816; PINC: 0.002587/0.003665/0.02047

## Solver verification
- Max abs difference between RK4 (dt = 1 s) and solve_ivp (DOP853, rtol = atol = 1e-11) over the test trajectories: 2.901e-09 K.

## Process parameters
ASSUMED / CALIBRATED parameters (see params.yaml for the sources):
- r1: 4.45153e-5   # ASSUMED
- r2: 6.88874e-9   # ASSUMED (per degC)
- s0: 0.33824      # ASSUMED
- s1: 0.0          # ASSUMED (the basic Ulleberg form has a constant s)
- s2: 0.0          # ASSUMED
- t1: -0.01539     # ASSUMED
- t2: 2.00181      # ASSUMED
- t3: 15.24178     # ASSUMED
- A: 0.25          # m^2   ASSUMED active cell area
- A_s: 1.48        # m^2   CALIBRATED loss area (hAs = h*A_s ~ 5.6 W/K)
- m0: 0.02         # kg/s  ASSUMED mass flow at valve opening u=1
- T_inc: 305.9     # K     CALIBRATED chiller supply temperature (a warm 33 degC coolant; cold-loop heat sink)

Calibrated nominal point (I = 40 A, u1 = 0.7, u2 = 0.5): T_out_ele = 319.99 K, T_in_ele = 314.99 K, T_out_c = 312.90 K, dT = 5.00 K, Vcell = 1.8696 V.
