import numpy as np

# Simulate Damage
# Number of defected Regions
damage_proportion = np.arange(0.05, 1.00, 0.05)
damage_proportion_bins = np.array([(damage_proportion[i], damage_proportion[i+1]) for i in range(0, len(damage_proportion)-1)])
def get_K(n, k0, dof=8, damage_proportion_bins=damage_proportion_bins):
    case = n
    k = k0*np.ones(dof)

    k_indices = np.arange(0, dof)
    damaged_k_indices = np.random.choice(k_indices, size=case, replace=False)
    damaged_k = []
    for i in damaged_k_indices:
        bin_indices = np.arange(0, len(damage_proportion_bins), 1)
        _bin = damage_proportion_bins[np.random.choice(bin_indices, size=1, replace=False)][0]
        damage = 1 - np.round(np.random.uniform(_bin[0], _bin[1]), decimals=3)
        damaged_k.append({
            'index': i,
            'damage' : damage,
            'value' : np.round(damage*k[i], decimals=3)
        })
        k[i] = damage*k[i] 

    K = np.zeros((dof, dof))
    for i in range(dof - 1):
        K[i, i] += k[i] + k[i + 1]
        K[i, i + 1] = -k[i + 1]
        K[i + 1, i] = -k[i + 1]
    K[dof - 1, dof - 1] = k[dof - 1]

    return {
        'K' : K,
        'k' : k,
        'damaged_indices': damaged_k_indices,
        'degree_of_damage' : n,
        'damaged_k_list' : damaged_k
    }

if __name__ == "__main__":
    print(get_K(2, 4))
