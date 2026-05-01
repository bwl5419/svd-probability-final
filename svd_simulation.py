## SVD rank selection Simulation
## Math 414h Final Paper

## Our goal is to use probability(random matirces + null distributions) to decide
## how many dimensions(r) to keep when doing SVD on a data matrix

## part 1: Build the null distrb to figure out what the singular values look like for pure boise
## part 2: False postiive calibraiton so if there is actually no singal, does my method still wrongly say there is a signal
## part 3: Can we actually detect real strcuture when it exists


import numpy as np  ## handles matrices and that
import matplotlib.pyplot as plt ## makes plots and such

def largest_singular_value(matrix):


    """
    Takes a 2D numpy matrix as our input

    Returns the single largest singular value as a float bc that's the only one we care about

    np.linalg.svd() does the full SVD decomposition.
    compute_uv=False means we only want the singular values and thus not U or V
    Singular values come back sorted largest to smallest, so our index [0] would be the biggest


    """
    singular_values= np.linalg.svd(matrix,compute_uv=False)
    return singular_values[0]


## now use out helper to estimate rank using the edge rule( which is just out way of choosing how many singular vlaues are "real")

def estimate_rank(matrix,threshold):
    """

    Takes a matrix and a threshold (which for us will be the 95th percentile cutoff).
    Returns r_hat: the number of singular values above the threshold.

    This is the core rank estimator from the paper:
        r_hat = # { k : s_k(X) > threshold }
        s_k(x) is the just the kth singular value
        so it's just the kth singular value is bigger than the cutoff that we chose
    """


    singular_values = np.linalg.svd(matrix,compute_uv=False)

    count = 0
    for sv in singular_values:
        if sv> threshold:
            count= count +1
    return count

## setting our matrix size and simulaiton count

m=20 ## our row count which is the number of dimensions
n=50 ## our colum count or our number of products
n_sim= 1000 ## this is the num of random matrices we're simulating

## part 1: Making the null distrb which is if x were pure noise, how large would that svd b
## so we're gonna simulate 1000 random matrices and record the largest sv each time

print("=" * 50) ## for organiaztion and making it clean looking
print("part 1:building the null distr")
print("=" * 50)

null_max_sv_list= [] ## this will have the largest sv from each sim

for i in range(n_sim):
    ## we gotta fen a random noise matrix
    ## thus each entry is drawn indepdently from N(0,1)
    # this is our prob model for noise

    E = np.random.randn(m,n)  # fix: randn for Gaussian N(0,1), not rand (uniform)

    ## now we get the largest singular value

    max_sv= largest_singular_value(E)

    null_max_sv_list.append(max_sv)

    ## we gotta conver the list to a numpy array so we can easily do the math on it

null_max_sv= np.array(null_max_sv_list)

    ## step 4 is to calc the 95th percentile so that 95% of ure noise matrices fall below this value
    ## thus if smth exceed this, it's prob real
q95= np.percentile(null_max_sv_list,95)

print(f"Mean of null max singular values:   {np.mean(null_max_sv):.4f}")
print(f"Std dev of null max singular values: {np.std(null_max_sv):.4f}")
print(f"95th percentile (our threshold q95): {q95:.4f}") ## 4f is a float with 4 digits
print()
## figure 1 -> histogram of null distr

plt.figure(figsize=(7,4))
plt.hist(null_max_sv_list, bins=30, color="#4C72B0", edgecolor="white")
plt.axvline(q95, color="crimson", linewidth=2, linestyle="--", label=f"95th percentile = {q95:.2f}")
plt.xlabel("largest singular value")
plt.ylabel("Frequency")
plt.title("Null Distribution of the Largest Singular Value")
plt.legend()
plt.tight_layout()  # fix: was missing () and newline before savefig
plt.savefig("figure1_null_distribution.png", dpi=150)
plt.show( )
print("Figure 1 saved: figure1_null_distribution.png")
print()

## part 2: false positive calibration
## does our threshold actuallly control false postives

print("="* 50)
print("PART 2: False positive calibration")
print("="* 50)

false_positive_list= []
## will store 1( a false positive) and 0(correct) for each sim

for i in range(n_sim):
    ## we want a pure noise matrix()
    E = np.random.randn(m,n)

    ## we get to etsimate rank using our threshold
    r_hat= estimate_rank(E,q95)

    ## if r_hat > 0, we made a false pos error  # fix: moved inside loop

    if r_hat > 0:  # fix: was outside for loop
        false_positive_list.append(1)
    else:
        false_positive_list.append(0)

false_positive_rate = sum(false_positive_list) /len(false_positive_list)

print(f"Number of false positives: {sum(false_positive_list)} out of {n_sim}")
print(f"Empirical false positive rate: {false_positive_rate:.3f}")
print(f"Expected (by construction):    0.050")
print()

## we want it to print smth close to 0.05, then our threshold will be well chosen

## Signal recovery

## can we actaully detect real structure
## so we plant a real rank 3 dignal and vary it's strenghts
## then we check how often we correctly guess r_hat

print("="* 50)

print("Part 3: Signal recovery(true rank =3)")
print("="*50)

true_rank = 3
## for 3 dimensions
spike_strength = [3,6,10] ## so we have weak, medium,strong

spike_labels= ["Weak(3)", "Medium(6)", "Strong(10)"]
mean_r_hat_list= []
## we;re gone store that mean estimted rank for each spike strenght

exact_recover_list = [] ## we store esact reovery rate for each spike strenght

for spike, label in zip(spike_strength, spike_labels):

    r_hat_list=[] ## estumated ranks for this spike strength

    for i in range(n_sim):
        ## build a random low rank signal matrix of true rank 3
        ## thne n.linalg.qr gives us random orthnomral matrices
        ## these are the directions in which the signal lives in
        U0, _ = np.linalg.qr(np.random.randn(m, true_rank))  # fix: UO (letter O) -> U0 (zero)
        V0, _ = np.linalg.qr(np.random.randn(n,true_rank)) ## fix: rand -> randn for Gaussian
        #m x3 orthnormal and nX3 orthnomal

        #l is the singal_spike controls how strong the signal is
        ## larger spike means it's easier to dectect above noise

        L = spike * (U0 @ V0.T) # fix: UO -> U0; we're doing matirx mult

        ## add the noise

        E=np.random.randn(m,n)  # fix: rand -> randn for Gaussian noise
        x = L + E ## the observed amtrix is the singal and the noise comboned

        ## esimate rank our null threshold

        r_hat= estimate_rank(x, q95)  # fix: was estimate_rank(m, n) -- passing dims not matrix
        r_hat_list.append(r_hat)

    ## sumarize for the spirk strength
    mean_r_hat = sum(r_hat_list)/ len(r_hat_list)

    #3 exact recover now whch will be how often did we exactly get r_hat=3

    exact_recovery= sum(1 for r in r_hat_list if r==true_rank)/ n_sim

    mean_r_hat_list.append(mean_r_hat)
    exact_recover_list.append(exact_recovery)

    print(f"Spike strength = {label}")
    print(f"  Mean estimated rank:  {mean_r_hat:.2f}  (true = {true_rank})")
    print(f"  Exact recovery rate:  {exact_recovery:.2f}  (want this close to 1.0)")
    print()

## figure 2, the bar chart of exact recover rates accros signal strenght  # fix: moved outside for loop

plt.figure(figsize=(6, 4))
bars = plt.bar(spike_labels, exact_recover_list, color=["#c6d8f0", "#7aaed6", "#2166ac"], edgecolor="white", width=0.5)
plt.axhline(1.0, color="black", linewidth=1, linestyle="--")
plt.text(len(spike_labels) - 0.5, 1.01, "Perfect recovery", ha="right", va="bottom", fontsize=9)
for bar, rate in zip(bars, exact_recover_list):
    plt.text(bar.get_x() + bar.get_width() / 2, rate + 0.01, f"{rate:.2f}", ha="center", va="bottom", fontsize=9)
plt.ylim(0, 1.15)

plt.xlabel("Signal Strength")
plt.ylabel("Exact Recovery Rate")
plt.title("Rank Recovery Rate by Signal Strength")
plt.tight_layout()
plt.savefig("figure2_recovery.png", dpi=150)
plt.show()
print("Figure 2 saved: figure2_recovery.png")
print()

print("=" * 50)
print("SIMULATION COMPLETE")
print("Use figure1 in Section 3 (null distribution)")
print("Use false positive rate in Section 3 (calibration)")
print("Use figure2 in Section 4 (signal recovery)")
print("=" * 50)
