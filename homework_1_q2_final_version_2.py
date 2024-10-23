import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
plt.close('all')


# Specific impulse values and delta_V array
Isp = 450 # Isp for each stage (s)
delta_V = np.array([0, 8, 2.46, 1.48, 0.68, 0.14, 0.68, 1.63, 0.68, 0.14, 3.14]) * 1000  # Delta V in m/s
#%% Question 2 
# Payload mass
m_payload = 1000  # kg
mpl = m_payload

def solve_for_Ri(speed, Isp):
    """Calculate the mass ratio Ri."""
    return np.exp(speed / (Isp * 9.80665))

# Create a list of maneuver names
maneuvers = ['NO NEED 0 ','NO NEED 1 ', '3. Transfer to GTO', '4. Orbit circularization in GEO', '5. Earth escape manuver from GTO', 
             '6. Midcourse  Correction', '7. Lunar Orbit Insertion (100 km)', 'int. Lunar Speed','8. Moon escape manuver', '9. Midcourse Correction 2 ', 
             '10. Transfer to LEO']

# Calculate R values for all maneuvers
R_values = []
for i in range(0, len(delta_V)):
    R = solve_for_Ri(delta_V[i], Isp)
    R_values.append(R)

# Print the result 
print("R values for each maneuver:")
for maneuver, R in zip(maneuvers, R_values):
    print(f"{maneuver}: {R:.4f}")
print()

# Mass ratios calculation
print('================')
multiplication_factor = 1  
for R in R_values[2:]:
    multiplication_factor *= R
        
overall_mass_ratio = multiplication_factor

# For the last stage 
ml_to_mo = 0.40
m0 = mpl/ml_to_mo
ms_mf = (1-ml_to_mo)*mpl/ml_to_mo

ms = m0/R_values[-1] - mpl
mf = m0-ms-mpl
m_initial = mf+ms+mpl
m_final = ms+mpl

# Tracing mass variation
mass_initial = []
mass_final = []

for i in range(len(R_values)-1, 1, -1):  # Start from the last stage, go backwards
    if i == len(R_values)-1:
        mass_final.append(m_final)
    else:
        mass_final.append(mass_initial[-1])
    
    mass_initial.append(mass_final[-1] * R_values[i])

# Reverse the lists to get chronological order
mass_initial = mass_initial[::-1]
mass_final = mass_final[::-1]

# Create a table of results
table_data = []
for i, (maneuver, mi, mf, R) in enumerate(zip(maneuvers[2:], mass_initial, mass_final, R_values[2:])):
    fuel_consumed = mi - mf
    table_data.append([maneuver, mi, mf, fuel_consumed, R])

# Add the R value column to the table headers
headers = ["Maneuver", "Initial Mass (kg)", "Final Mass (kg)", "Fuel Consumed (kg)", "R (Mass Ratio)"]
print(tabulate(table_data, headers=headers, floatfmt=".2f"))

# Extract maneuver names and initial masses from table_data
maneuver_names = [row[0].split('. ')[-1] for row in table_data]
initial_masses = [row[1] for row in table_data]

# Add "final" state with the smallest value from FINAL MASS
final_mass = min(row[2] for row in table_data)
maneuver_names.append("Final")
initial_masses.append(final_mass)

plt.figure(figsize=(14, 8))
plt.plot(range(len(initial_masses)), initial_masses, marker='o', linestyle='-', linewidth=2, markersize=8)

plt.title('Initial Mass Variation for Each Maneuver (Original Order)', fontsize=16)
plt.xlabel('Maneuver', fontsize=12)
plt.ylabel('Mass (kg)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Set x-ticks to maneuver names
plt.xticks(range(len(maneuver_names)), maneuver_names, rotation=45, ha='right')

# Add value labels
for i, txt in enumerate(initial_masses):
    plt.annotate(f'{txt:.2f}', (i, txt), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=9)

plt.tight_layout()
plt.show()
