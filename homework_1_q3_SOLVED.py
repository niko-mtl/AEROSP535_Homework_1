import numpy as np
from scipy.interpolate import interp1d
import itertools

mass_upper_stage = 14942.26

# Create new delta_v range with 200 steps
delta_v_values = np.arange(600, 8200, 200)  # Goes up to 8000 inclusive

# Original data points
original_delta_v = np.array([600, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])
original_epsilon_kerosene = np.array([0.078, 0.060, 0.042, 0.038, 0.035, 0.032, 0.030, 0.030, 0.030])
original_epsilon_methane = np.array([0.080, 0.060, 0.042, 0.038, 0.035, 0.032, 0.030, 0.030, 0.030])
original_epsilon_hydrogen = np.array([np.nan, 0.14, 0.10, 0.082, 0.078, 0.075, 0.073, 0.070, 0.068])

# Create interpolation functions
f_kerosene = interp1d(original_delta_v, original_epsilon_kerosene, kind='nearest')
f_methane = interp1d(original_delta_v, original_epsilon_methane, kind='nearest')
# For hydrogen, we'll start from 1000 since we don't have a value at 600
f_hydrogen = interp1d(original_delta_v[1:], original_epsilon_hydrogen[1:], kind='nearest')

# Generate interpolated values
epsilon_kerosene = np.round(f_kerosene(delta_v_values), decimals=4)
epsilon_methane = np.round(f_methane(delta_v_values), decimals=4)

# For hydrogen, handle None values separately
hydrogen_values = np.zeros(len(delta_v_values))
hydrogen_mask = delta_v_values < 1000
hydrogen_values[~hydrogen_mask] = f_hydrogen(delta_v_values[~hydrogen_mask])
hydrogen_values = np.round(hydrogen_values, decimals=4)
epsilon_hydrogen = np.where(hydrogen_mask, None, hydrogen_values)

# Convert to regular Python lists for easier use in your code
delta_v_values = list(delta_v_values)
epsilon_kerosene = list(epsilon_kerosene)
epsilon_methane = list(epsilon_methane)
epsilon_hydrogen = list(epsilon_hydrogen)

print("delta_v_values =", delta_v_values)
print("\nepsilon_kerosene =", epsilon_kerosene)
print("\nepsilon_methane =", epsilon_methane)
print("\nepsilon_hydrogen =", epsilon_hydrogen)

ISP_kerosene = 280
ISP_methane = 350
ISP_hydrogen = 450
g0 = 9.80665

# Map fuels to their ISP values
isp_values = {
    'kerosene': ISP_kerosene,
    'methane': ISP_methane,
    'hydrogen': ISP_hydrogen
}

# Define the total delta_V
total_delta_v = 8200

# Generate all possible combinations of 3 stages where the sum is 8000 using only values from delta_v_values
possible_combinations = [
    combo for combo in itertools.product(delta_v_values, repeat=3)
    if sum(combo) == total_delta_v
]

# Define fuel types and their corresponding epsilon values
fuels = ['kerosene', 'methane', 'hydrogen']
epsilon_values = {
    'kerosene': epsilon_kerosene,
    'methane': epsilon_methane,
    'hydrogen': epsilon_hydrogen
}

# Generate all possible fuel assignments for the 3 stages
fuel_combinations = list(itertools.product(fuels, repeat=3))

# Store the results
results = []

# Iterate over all delta_V combinations and fuel combinations
for delta_v_combo in possible_combinations:
    for fuel_combo in fuel_combinations:
        valid_combination = True
        stage_info = []
        
        # Collect info for each stage
        for i, delta_v in enumerate(delta_v_combo):
            fuel = fuel_combo[i]
            index = delta_v_values.index(delta_v)  # Get the index of delta_v
            epsilon = epsilon_values[fuel][index]  # Get epsilon for this fuel and delta_v
            isp = isp_values[fuel]  # Get ISP for this fuel
            
            if epsilon is None:  # Skip invalid combinations
                valid_combination = False
                break
            
            stage_info.append((delta_v, fuel, epsilon, isp))
        
        # If the combination is valid, store the result
        if valid_combination:
            results.append(stage_info)

# Display results
for result in results:
    print(f"Stages: {result}")

# Example of accessing first result
if results:
    result = results[0]
    delta_V1 = result[0][0]
    delta_V2 = result[1][0]
    delta_V3 = result[2][0]
    print("\nFirst result details:")
    print(result)
    print(f"Delta V1: {delta_V1}")
    print(f"Delta V2: {delta_V2}")
    print(f"Delta V3: {delta_V3}")

#%%
import numpy as np
import itertools
from scipy.optimize import fsolve


results_new = []
for result in results:
    delta_V1 = result[0][0]
    name_1 = (result[0][1])
    name_2 = (result[1][1])
    name_3 = (result[2][1])
    if name_1==name_2==name_3:
        False
    else :
        results_new.append(result) 
    
results = results_new 

# New list to store results with R values
results_with_R = []

for result in results:
    delta_V1 = result[0][0]
    delta_V2 = result[1][0]
    delta_V3 = result[2][0]
    
    epsilon_1 = result[0][2]
    epsilon_2 = result[1][2]
    epsilon_3 = result[2][2]
    
    ISP_1 = result[0][3]
    ISP_2 = result[1][3]
    ISP_3 = result[2][3]
    
    # Define the function to solve for alpha
    def equation(alpha):
        R1 = (alpha * ISP_1 * g0 + 1)/(alpha * ISP_1 * g0 * epsilon_1)
        R2 = (alpha * ISP_2 * g0 + 1)/(alpha * ISP_2 * g0 * epsilon_2)
        R3 = (alpha * ISP_3 * g0 + 1)/(alpha * ISP_3 * g0 * epsilon_3)
        return 8200 - (ISP_1 * g0 * np.log(R1) + ISP_2 * g0 * np.log(R2) + ISP_3 * g0 * np.log(R3))
    
    # Solve for alpha with initial guess
    alpha = fsolve(equation, -4e-4)[0]
    
    # Calculate R1, R2, R3
    R1 = (alpha * ISP_1 * g0 + 1)/(alpha * ISP_1 * g0 * epsilon_1)
    R2 = (alpha * ISP_2 * g0 + 1)/(alpha * ISP_2 * g0 * epsilon_2)
    R3 = (alpha * ISP_3 * g0 + 1)/(alpha * ISP_3 * g0 * epsilon_3)

    
    lambda1 = (1-R1*epsilon_1)/(R1-1)
    lambda2 = (1-R2*epsilon_2)/(R2-1)
    lambda3 = (1-R3*epsilon_3)/(R3-1)
    
    mo3 = mass_upper_stage*(lambda3+1)/lambda3
    mo2 = mo3*(lambda2+1)/lambda2
    mo1 = mo2*(lambda1+1)/lambda1
    
    m_total = mo1
    # Create new result with R values
    new_result = [
        (result[0][0], result[0][1], result[0][2], result[0][3], round(R1,2), round(lambda1,2),round(mo1) ),
        (result[1][0], result[1][1], result[1][2], result[1][3], round(R2,2), round(lambda2,2),round(mo2) ),
        (result[2][0], result[2][1], result[2][2], result[2][3], round(R3,2), round(lambda3,2),round(mo3) ),
        round(m_total)
    ]
    results_with_R.append(new_result)
    print(new_result)

# Replace original results with new results
results = results_with_R
#%%
res =[]
for result in results:
    print(result)
    
    v1 = result[0][0] 
    v2 = result[1][0]
    v3 = result[2][0]
    
    R1 = result[0][4]
    R2 = result[1][4]
    R3 = result[2][4]
    
    lambda1 = result[0][5]
    lambda2 = result[1][5]
    lambda3 = result[2][5]
    
    if (R1>1 and R2>1 and R3>1) and (lambda1<1 and lambda2<1 and lambda3<1) and (v1>v2>v3) :
        res.append(result)
    
res.sort(key=lambda x: x[3],reverse = False)

print('-----------')
for i in range(10):
    print(i)
    print(res[i])
    print()
print("My choice is: i = 0")
print(res[0])
from scipy.optimize import fsolve

def calculate_alpha_for_result(result):
    # Extract delta_V, epsilon, and ISP values from the chosen result
    delta_V1 = result[0][0]
    delta_V2 = result[1][0]
    delta_V3 = result[2][0]

    epsilon_1 = result[0][2]
    epsilon_2 = result[1][2]
    epsilon_3 = result[2][2]

    ISP_1 = result[0][3]
    ISP_2 = result[1][3]
    ISP_3 = result[2][3]

    # Define the equation to solve for alpha
    def equation(alpha):
        R1 = (alpha * ISP_1 * g0 + 1)/(alpha * ISP_1 * g0 * epsilon_1)
        R2 = (alpha * ISP_2 * g0 + 1)/(alpha * ISP_2 * g0 * epsilon_2)
        R3 = (alpha * ISP_3 * g0 + 1)/(alpha * ISP_3 * g0 * epsilon_3)
        return 8200 - (ISP_1 * g0 * np.log(R1) + ISP_2 * g0 * np.log(R2) + ISP_3 * g0 * np.log(R3))

    # Solve for alpha using initial guess
    alpha = fsolve(equation, -4e-4)[0]

    print(f"The alpha for the chosen result is: {alpha:.6f}")

# Example of calling the function for your chosen result (i=5):
calculate_alpha_for_result(res[0])
